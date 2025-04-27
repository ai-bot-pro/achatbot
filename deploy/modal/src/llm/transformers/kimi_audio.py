from time import perf_counter
import time
from typing import Optional
import modal
import os

app = modal.App("kimi_audio")
kimi_audio_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/Kimi-Audio.git",
        "cd /Kimi-Audio && git submodule update --init --recursive",
        "cd /Kimi-Audio && pip install -q -r requirements.txt",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "cd /Kimi-Audio && git pull origin feat/achatbot",
        "cd /Kimi-Audio && git checkout 05811bffb7221f60fb7ff2435754779218d2c219",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

with kimi_audio_img.imports():
    import os
    import sys
    import subprocess
    import torch

    sys.path.insert(0, "/Kimi-Audio")
    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.api.prompt_manager import KimiAPromptManager
    from kimia_infer.utils.sampler import KimiASampler

    model_path = os.path.join(HF_MODEL_DIR, "moonshotai/Kimi-Audio-7B-Instruct")
    model = KimiAudio(model_path=model_path, load_detokenizer=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40s"),
    cpu=2.0,
    retries=0,
    image=kimi_audio_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func()


def tokenize():
    prompt_manager = KimiAPromptManager(
        model_path=model_path,
        kimia_token_offset=152064,
    )
    print("extra_tokens", prompt_manager.extra_tokens)

    messages_asr = [
        # You can provide context or instructions as text
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the following audio:",
        },
        # Provide the audio file path
        {
            "role": "user",
            "message_type": "audio",
            "content": "/Kimi-Audio/test_audios/asr_example.wav",
        },
    ]
    asr_prompt = prompt_manager.get_prompt(messages_asr, output_type="text")
    print("asr_prompt", asr_prompt)

    messages_conversation = [
        # Start conversation with an audio query
        {
            "role": "user",
            "message_type": "audio",
            "content": "/Kimi-Audio/test_audios/qa_example.wav",
        }
    ]
    conversation_prompt = prompt_manager.get_prompt(messages_conversation, output_type="both")
    print("conversation_prompt", conversation_prompt)


def asr():
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    messages_asr = [
        # You can provide context or instructions as text
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the following audio:",
        },
        # Provide the audio file path
        {
            "role": "user",
            "message_type": "audio",
            "content": "/Kimi-Audio/test_audios/asr_example.wav",
        },
    ]

    # Generate only text output
    _, text_output = model.generate(messages_asr, **sampling_params, output_type="text")
    print(
        ">>> ASR Output Text: ", text_output
    )  # Expected output: "这并不是告别，这是一个篇章的结束，也是新篇章的开始。"


def conversation():
    import soundfile as sf

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    messages_conversation = [
        # Start conversation with an audio query
        {
            "role": "user",
            "message_type": "audio",
            "content": "/Kimi-Audio/test_audios/qa_example.wav",
        }
    ]

    # Generate both audio and text output
    wav_output, text_output = model.generate(
        messages_conversation, **sampling_params, output_type="both"
    )

    # Save the generated audio
    output_audio_path = os.path.join(ASSETS_DIR, "output_audio.wav")
    sf.write(
        output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000
    )  # Assuming 24kHz output
    print(f">>> Conversational Output Audio saved to: {output_audio_path}")
    print(">>> Conversational Output Text: ", text_output)  # Expected output: "A."

    print("Kimi-Audio inference examples complete.")


def asr_stream():
    import torch

    model = KimiAudio(model_path=model_path, load_detokenizer=False)

    messages_asr = [
        # You can provide context or instructions as text
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the following audio:",
        },
        # Provide the audio file path
        {
            "role": "user",
            "message_type": "audio",
            "content": "/Kimi-Audio/test_audios/asr_example.wav",
        },
    ]

    history = model.prompt_manager.get_prompt(messages_asr, output_type="text")
    audio_input_ids, text_input_ids, is_continuous_mask = history.to_tensor()
    audio_continuous_features = history.continuous_feature
    print(
        "input_token_ids:",
        audio_input_ids.shape,
        text_input_ids.shape,
        is_continuous_mask.shape,
        len(audio_continuous_features),
        audio_continuous_features[0].shape,
    )

    audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
    text_input_ids = text_input_ids.to(torch.cuda.current_device())
    is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
    audio_continuous_features = [
        f.to(torch.cuda.current_device()) for f in audio_continuous_features
    ]

    max_new_tokens = 7500 - audio_input_ids.shape[1]
    sampler = KimiASampler(
        audio_top_k=10,
        audio_temperature=0.8,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_top_k=5,
        text_temperature=0.0,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
    )

    for chunk in gen_chunk_stream(
        audio_input_ids,
        sampler,
        text_input_ids,
        is_continuous_mask,
        audio_continuous_features,
        output_type="text",
        max_new_tokens=max_new_tokens,
    ):
        print(chunk)


def gen_chunk_stream(
    audio_input_ids: torch.Tensor,  # input audio tokens
    sampler: "KimiASampler",
    text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
    is_continuous_mask: torch.Tensor = None,
    audio_continous_features: torch.Tensor = None,
    output_type: str = "text",
    max_new_tokens: int = 50,
):
    text_stream_is_finished = False
    previous_audio_tokens = torch.zeros(
        (4096,),
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )
    text_previous_tokens = torch.zeros(
        (4096,),
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )

    decoder_input_audio_ids = audio_input_ids.clone()
    decoder_input_text_ids = text_input_ids.clone()
    decoder_position_ids = (
        torch.arange(0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device())
        .unsqueeze(0)
        .long()
    )
    decoder_input_whisper_feature = audio_continous_features
    decoder_is_continuous_mask = is_continuous_mask
    past_key_values = None

    last_position_id = decoder_input_audio_ids.shape[1] - 1

    valid_text_length = 0
    valid_audio_length = 0

    audio_logits, text_logits, past_key_values = model.alm.forward(
        input_ids=decoder_input_audio_ids,
        text_input_ids=decoder_input_text_ids,
        whisper_input_feature=decoder_input_whisper_feature,
        is_continuous_mask=decoder_is_continuous_mask,
        position_ids=decoder_position_ids,
        past_key_values=past_key_values,
        return_dict=False,
    )


def detokenize_audio_stream(self, vq_codes: torch.Tensor) -> torch.Tensor:
    if self.detokenizer is None:
        raise ValueError("Detokenizer is not initialized")
    self.detokenizer.clear_states()
    chunk_size = 30  # hard-coded right now
    first_chunk_size = 30
    cache_speech_collection = []
    audio_tokens = vq_codes.to(torch.cuda.current_device())
    audio_tokens = audio_tokens.long()
    num_audio_tokens = audio_tokens.size(1)
    first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
    gen_speech = self.detokenizer.detokenize_streaming(
        first_chunk_semantic_tokens,
        is_final=(num_audio_tokens <= first_chunk_size),
        upsample_factor=4,
    )
    cache_speech_collection.append(gen_speech)

    if num_audio_tokens > first_chunk_size:
        res_semantic_tokens = audio_tokens[:, first_chunk_size:]
        for i in range(0, res_semantic_tokens.size(1), chunk_size):
            chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
            gen_speech = self.detokenizer.detokenize_streaming(
                chunk_semantic_tokens,
                upsample_factor=4,
                is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
            )
            cache_speech_collection.append(gen_speech)

    gen_speech = torch.cat(cache_speech_collection, dim=-1)
    return gen_speech


"""
IMAGE_GPU=L4 modal run src/llm/transformers/kimi_audio.py --task tokenize

"""


@app.local_entrypoint()
def main(task: str = "tokenize"):
    tasks = {
        "tokenize": tokenize,
        "asr": asr,
        "conversation": conversation,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
