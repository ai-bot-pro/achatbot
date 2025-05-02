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
        "cd /Kimi-Audio && git checkout e6ef56fd7466feae8d4ea4341631e23dc5e63853",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "ALIAS_FREE_ACTIVATION_CUDA_SRC_PATH": "/Kimi-Audio/kimia_infer/models/detokenizer/vocoder/alias_free_activation/cuda",
            "MEL_FILTERS_PATH": "/Kimi-Audio/kimia_infer/models/tokenizer/whisper_Lv3/mel_filters.npz",
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

    # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/tokenization_kimia.py#L141
    token_ids = prompt_manager.text_tokenizer.encode("你叫什么名字？", bos=True, eos=True)
    print("text_token_ids:", token_ids)

    messages_cases = {
        "text": [
            {
                "role": "user",
                "message_type": "text",
                "content": "你叫什么名字？",
            },
        ],
        "audio": [
            {
                "role": "user",
                "message_type": "audio",
                "content": "/Kimi-Audio/test_audios/qa_example.wav",
            }
        ],
        "text_audio": [
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
        ],
    }
    for case_name, messages in messages_cases.items():
        print(10 * "---" + case_name + 10 * "---")
        prompt = prompt_manager.get_prompt(messages, output_type="both")
        print(prompt)

        text_token = prompt_manager.text_tokenizer.decode(prompt.text_token_ids)
        print(f"text_token: {text_token}")

        # NOTE: audio token id > kimia_token_offset not supported in text tokenizer map dict
        # need use add_special_tokens to add special tokens
        audio_token = ""
        for tid in prompt.audio_token_ids:
            if tid < prompt_manager.kimia_token_offset:
                audio_token += prompt_manager.text_tokenizer.decode([tid])
            else:
                audio_token += f"<|audio_token_{tid}|>"
        print(f"audio_token: {audio_token}")


def asr():
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
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
    """
    A1->T2A2
    """
    import soundfile as sf

    model = KimiAudio(model_path=model_path, load_detokenizer=True)

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
    output_audio_path = os.path.join(ASSETS_DIR, "kimia_conversation.wav")
    sf.write(
        output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000
    )  # Assuming 24kHz output
    print(f">>> Conversational Output Audio saved to: {output_audio_path}")
    print(">>> Conversational Output Text: ", text_output)


def conversation_text_audio():
    """
    T1->T2A2
    """
    import soundfile as sf

    model = KimiAudio(model_path=model_path, load_detokenizer=True)

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
            "message_type": "text",
            "content": "你叫什么名字?",
        }
    ]

    # Generate both audio and text output
    wav_output, text_output = model.generate(
        messages_conversation, **sampling_params, output_type="both"
    )

    # Save the generated audio
    output_audio_path = os.path.join(ASSETS_DIR, "kimia_conversation_text_audio.wav")
    sf.write(
        output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000
    )  # Assuming 24kHz output
    print(f">>> Conversational Output Audio saved to: {output_audio_path}")
    print(">>> Conversational Output Text: ", text_output)


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

    prompt = model.prompt_manager.get_prompt(messages_asr, output_type="text")
    audio_input_ids, text_input_ids, is_continuous_mask = prompt.to_tensor()
    audio_continuous_features = prompt.continuous_feature
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

    times = []
    start_time = time.perf_counter()
    for text_token_id, _ in gen_token_stream(
        model,
        audio_input_ids,
        sampler,
        text_input_ids,
        is_continuous_mask,
        audio_continuous_features,
        output_type="text",
        max_new_tokens=max_new_tokens,
    ):
        times.append(time.perf_counter() - start_time)
        # print(text_token_id)
        if text_token_id.item() == model.extra_tokens.kimia_text_eos:
            break
        if text_token_id.item() == model.extra_tokens.kimia_text_blank:
            print(" ", end="", flush=True)
            continue
        print(
            model.prompt_manager.text_tokenizer.decode([text_token_id.item()]), end="", flush=True
        )
        start_time = time.perf_counter()
    print(f"TTFT: {times[0]} s | total: {sum(times)} s | avg: {sum(times)/len(times)} s")


def conversation_stream():
    import soundfile as sf

    model = KimiAudio(model_path=model_path, load_detokenizer=True)

    output_type = "both"
    messages_conversation_cases = {
        "kimia_conversation_T1toT2A2": [
            {
                "role": "user",
                "message_type": "text",
                "content": "你叫什么名字?",
            }
        ],
        "kimia_conversation_A1toT2A2": [
            {
                "role": "user",
                "message_type": "audio",
                "content": "/Kimi-Audio/test_audios/qa_example.wav",
            }
        ],
    }
    for case_name, messages_conversation in messages_conversation_cases.items():
        prompt = model.prompt_manager.get_prompt(messages_conversation, output_type=output_type)
        audio_input_ids, text_input_ids, is_continuous_mask = prompt.to_tensor()
        audio_continuous_features = prompt.continuous_feature
        print(
            "input_token_ids:",
            audio_input_ids.shape,
            text_input_ids.shape,
            is_continuous_mask.shape,
            len(audio_continuous_features),
            audio_continuous_features[0].shape if len(audio_continuous_features) > 0 else None,
        )

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_continuous_features = [
            f.to(torch.cuda.current_device()) for f in audio_continuous_features
        ]

        max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        print(f"max_new_tokens: {max_new_tokens}")
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

        audio_text = ""
        audio_vq_codes = []
        gen_speechs = []
        times = []
        audio_chunk_times = []
        start_time = time.perf_counter()
        for text_token_id, audio_token_id in gen_token_stream(
            model,
            audio_input_ids,
            sampler,
            text_input_ids,
            is_continuous_mask,
            audio_continuous_features,
            output_type=output_type,
            max_new_tokens=max_new_tokens,
        ):
            times.append(time.perf_counter() - start_time)

            text_token_id = text_token_id.item()
            audio_token_id = audio_token_id.item()

            if output_type == "text" and text_token_id == model.extra_tokens.kimia_text_eos:
                break

            if (
                text_token_id != model.extra_tokens.kimia_text_eos
                and text_token_id < model.kimia_token_offset
                and text_token_id != model.extra_tokens.kimia_text_blank
            ):
                # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/tokenization_kimia.py
                # no skip_special_tokens args :)
                audio_text += model.prompt_manager.text_tokenizer.decode([text_token_id])

            if output_type == "text":
                continue

            if audio_token_id < model.kimia_token_offset:
                continue
            if audio_token_id in model.eod_ids:
                break

            audio_vq_code = audio_token_id - model.kimia_token_offset
            audio_vq_codes.append(audio_vq_code)
            if len(audio_vq_codes) % 30 == 0:
                print("internal_vq_codes", audio_vq_codes)
                start_time = time.perf_counter()
                gen_speech = model.detokenizer.detokenize_streaming(
                    torch.tensor(audio_vq_codes)
                    .unsqueeze(0)
                    .long()
                    .to(torch.cuda.current_device()),
                    is_final=False,
                    upsample_factor=4,
                )
                audio_chunk_times.append(time.perf_counter() - start_time)
                audio_vq_codes = []
                gen_speechs.append(gen_speech)

                output_path = os.path.join(ASSETS_DIR, f"{case_name}_{len(gen_speechs)}.wav")
                sf.write(
                    output_path,
                    gen_speech.detach().cpu().view(-1).numpy(),
                    24000,
                )

            start_time = time.perf_counter()

        if len(audio_vq_codes) > 0:
            print("last_vq_codes", audio_vq_codes)
            start_time = time.perf_counter()
            gen_speech = model.detokenizer.detokenize_streaming(
                torch.tensor(audio_vq_codes).unsqueeze(0).long().to(torch.cuda.current_device()),
                is_final=True,
                upsample_factor=4,
            )
            audio_chunk_times.append(time.perf_counter() - start_time)
            gen_speechs.append(gen_speech)

            output_path = os.path.join(ASSETS_DIR, f"{case_name}_{len(gen_speechs)}.wav")
            sf.write(
                output_path,
                gen_speech.detach().cpu().view(-1).numpy(),
                24000,
            )

        print(
            f"text TTFT: {times[0]} s | total: {sum(times)} s | len: {len(times)} | avg: {sum(times)/len(times)} s"
        )

        print(
            f"audio TTFT(chunk): {audio_chunk_times[0]} s | total: {sum(audio_chunk_times)} s | len: {len(audio_chunk_times)} | avg: {sum(audio_chunk_times)/len(audio_chunk_times)} s"
        )

        output_audio_path = os.path.join(ASSETS_DIR, f"{case_name}.wav")
        sf.write(
            output_audio_path, torch.cat(gen_speechs, dim=-1).detach().cpu().view(-1).numpy(), 24000
        )  # Assuming 24kHz output
        print(f">>> Conversational Output Audio saved to: {output_audio_path}")
        print(">>> Conversational Output Text: ", audio_text)


def gen_token_stream(
    model: "KimiAudio",
    audio_input_ids: torch.Tensor,  # input audio tokens
    sampler: "KimiASampler",
    text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
    is_continuous_mask: torch.Tensor = None,
    audio_continous_features: torch.Tensor = None,
    output_type: str = "text",
    max_new_tokens: int = 50,
):
    assert output_type in ["text", "audio", "both"], f"output_type: {output_type}"

    is_output_audio = output_type == "both" or output_type == "audio"

    text_stream_is_finished = False
    previous_audio_tokens = torch.zeros(
        (max_new_tokens,),
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )
    text_previous_tokens = torch.zeros(
        (max_new_tokens,),
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
    decoder_input_whisper_feature = audio_continous_features or None
    decoder_is_continuous_mask = is_continuous_mask
    past_key_values = None

    last_position_id = decoder_input_audio_ids.shape[1] - 1

    # one bye one generate, until eos or max_new_tokens
    for i in range(max_new_tokens):
        # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/modeling_moonshot_kimia.py#L850
        # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/config.json
        # use_cache=True
        audio_logits, text_logits, past_key_values = model.alm.forward(
            input_ids=decoder_input_audio_ids,
            text_input_ids=decoder_input_text_ids,
            whisper_input_feature=decoder_input_whisper_feature,
            is_continuous_mask=decoder_is_continuous_mask,
            position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            return_dict=False,
        )

        # Sample text token using the sampler
        next_text_token_id = sampler.sample_text_logits(
            text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
        )
        print(f"{i} next_text_token_id", next_text_token_id)
        # Sample audio token using the sampler
        next_audio_token_id = (
            sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )
            if i >= model.kimia_text_audiodelaytokens and is_output_audio
            else torch.Tensor([model.extra_tokens.kimia_text_blank])
            .long()
            .to(torch.cuda.current_device())
        )
        print(f"{i} next_audio_token_id", next_audio_token_id)

        if text_stream_is_finished:
            next_text_token_id.fill_(model.extra_tokens.kimia_text_blank)
        elif next_text_token_id.item() == model.extra_tokens.kimia_text_eos:
            text_stream_is_finished = True
        audio_stream_is_finished = next_audio_token_id.item() in model.eod_ids

        yield (next_text_token_id, next_audio_token_id)  # (1,) (1,)

        if output_type == "text" and text_stream_is_finished:
            break
        if output_type == "both" and text_stream_is_finished and audio_stream_is_finished:
            break

        text_previous_tokens[i : i + 1] = next_text_token_id
        previous_audio_tokens[i : i + 1] = next_audio_token_id

        decoder_input_audio_ids = next_audio_token_id.unsqueeze(1)  # (1,1)
        decoder_input_text_ids = next_text_token_id.unsqueeze(1)  # (1,1)
        last_position_id += 1
        decoder_position_ids = (
            torch.zeros(1, 1, device=torch.cuda.current_device())
            .fill_(last_position_id)
            .long()
            .view(1, 1)
        )  # (1,1)
        decoder_input_whisper_feature = None
        decoder_is_continuous_mask = None


"""
IMAGE_GPU=L4 modal run src/llm/transformers/kimi_audio.py --task tokenize

IMAGE_GPU=L40s modal run src/llm/transformers/kimi_audio.py --task asr
IMAGE_GPU=L40s modal run src/llm/transformers/kimi_audio.py --task asr_stream

IMAGE_GPU=L40s modal run src/llm/transformers/kimi_audio.py --task conversation
IMAGE_GPU=L40s modal run src/llm/transformers/kimi_audio.py --task conversation_stream
IMAGE_GPU=L40s modal run src/llm/transformers/kimi_audio.py --task conversation_text_audio
"""


@app.local_entrypoint()
def main(task: str = "tokenize"):
    tasks = {
        "tokenize": tokenize,
        "asr": asr,
        "asr_stream": asr_stream,
        "conversation": conversation,
        "conversation_stream": conversation_stream,
        "conversation_text_audio": conversation_text_audio,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
