import modal
import os

app = modal.App("qwen2_5_omni")
tag_or_commit = os.getenv("TAG_OR_COMMIT", "21dbefaa54e5bf180464696aa70af0bfc7a61d53")
omni_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .run_commands(
        f"pip install git+https://github.com/huggingface/transformers@{tag_or_commit}",
    )
    .pip_install(
        "accelerate",
        "torch",
        "torchvision",
        "torchaudio",
        "soundfile==0.13.0",
        "librosa==0.11.0",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

# NOTE: if want to generate speech, need use this system prompt to generate speech
SPEECH_SYS_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
# Voice settings
SPEAKER_LIST = ["Chelsie", "Ethan"]
DEFAULT_SPEAKER = "Ethan"

with omni_img.imports():
    import torch, torchaudio
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    import subprocess

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    model_path = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()

    # NOTE: when disable talker, generate must set return_audio=False
    # model.disable_talker()

    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    # print(model)
    print(f"{model_million_params} M parameters")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    subprocess.run("nvidia-smi", shell=True)

    def inference(
        messages,
        return_audio=False,
        use_audio_in_video=False,
        thinker_do_sample=False,
        speaker=DEFAULT_SPEAKER,
    ):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info([messages])
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio,
            speaker=speaker,
            thinker_do_sample=thinker_do_sample,
        )
        print("\n====generate use memory=====\n")
        subprocess.run(
            """nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{print "GPU "$1": "$2"/"$3" MiB\\n"}'""",
            shell=True,
        )
        print("\n=========\n")
        # print(output)
        text_token_ids = output
        audio = None
        if return_audio and len(output) > 1:
            text_token_ids = output[0].detach()
            audio = output[1].unsqueeze(0).detach()

        text = processor.batch_decode(
            text_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        torch.cuda.empty_cache()

        return text, audio


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40s"),
    cpu=2.0,
    retries=0,
    image=omni_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run(func):
    func()


def voice_chatting():
    sys_msg = {
        "role": "system",
        "content": SPEECH_SYS_PROMPT,
    }

    for audio_path in ["guess_age_gender.wav", "translate_to_chinese.wav"]:
        audio_path = os.path.join(ASSETS_DIR, audio_path)
        audio_msg = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        texts, audio = inference(audio_msg, return_audio=True, use_audio_in_video=True)
        print(texts[0], audio.shape)

        save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(audio_path)}")
        torchaudio.save(save_audio_path, audio, sample_rate=24000)
        print(f"Audio saved to {save_audio_path}")


def multi_round_omni_chatting():
    conversations = [
        {
            "role": "system",
            "content": SPEECH_SYS_PROMPT,
        },
    ]
    for video_path in ["draw1.mp4", "draw2.mp4", "draw3.mp4"]:
        video_path = os.path.join(ASSETS_DIR, video_path)
        conversations.append({"role": "user", "content": [{"type": "video", "video": video_path}]})
        texts, audio = inference(conversations, return_audio=True, use_audio_in_video=True)
        print(texts[0], audio.shape)
        save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
        torchaudio.save(save_audio_path, audio, sample_rate=24000)
        print(f"Audio saved to {save_audio_path}")


def omni_chatting_for_math():
    video_path = os.path.join(ASSETS_DIR, "math.mp4")
    messages = [
        {
            "role": "system",
            "content": SPEECH_SYS_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
            ],
        },
    ]

    response, audio = inference(messages, return_audio=True, use_audio_in_video=True)
    print(response[0], audio.shape)

    save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
    torchaudio.save(save_audio_path, audio, sample_rate=24000)
    print(f"Audio saved to {save_audio_path}")


def omni_chatting_for_music():
    video_path = os.path.join(ASSETS_DIR, "music.mp4")
    messages = [
        {
            "role": "system",
            "content": SPEECH_SYS_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
            ],
        },
    ]
    response, audio = inference(messages, return_audio=True, use_audio_in_video=True)
    print(response[0])

    save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
    torchaudio.save(save_audio_path, audio, sample_rate=24000)
    print(f"Audio saved to {save_audio_path}")


def screen_recording_interaction():
    video_path = os.path.join(ASSETS_DIR, "screen.mp4")
    for prompt in [
        "What the browser is used in this video?",
        "浏览器中的论文叫什么名字?",
        "这篇论文主要解决什么问题呢？",
    ]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            },
        ]

        response, _ = inference(messages, return_audio=False, use_audio_in_video=False)
        print(response[0])


def universal_audio_understanding():
    for case in [
        {
            "audio_path": "1272-128104-0000.flac",
            "prompt": "Transcribe the English audio into text without any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "BAC009S0764W0121.wav",
            "prompt": "请将这段中文语音转换为纯文本，去掉标点符号。",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "10000611681338527501.wav",
            "prompt": "Transcribe the Russian audio into text without including any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "7105431834829365765.wav",
            "prompt": "Transcribe the French audio into text without including any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "1272-128104-0000.flac",
            "prompt": "Listen to the provided English speech and produce a translation in Chinese text.",
            "sys_prompt": "You are a speech translation model.",
        },
        {
            "audio_path": "cough.wav",
            "prompt": "Classify the given human vocal sound in English.",
            "sys_prompt": "You are a voice classification model.",
        },
    ]:
        audio_path = os.path.join(ASSETS_DIR, case["audio_path"])
        messages = [
            {"role": "system", "content": case["sys_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case["prompt"]},
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]
        texts, _ = inference(messages, use_audio_in_video=True, return_audio=False)
        print(texts[0])


def video_information_extracting():
    video_path = os.path.join(ASSETS_DIR, "shopping.mp4")
    sys_msg = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    for prompt in [
        "How many kind of drinks can you see in the video?",
        "How many bottles of drinks have I picked up?",
        "How many milliliters are there in the bottle I picked up second time?",
        "视屏中的饮料叫什么名字呢？",
        "跑步🏃🏻累了，适合喝什么饮料补充体力呢？",
    ]:
        messages = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            },
        ]
        texts, _ = inference(messages, return_audio=False, use_audio_in_video=False)
        print(texts[0])


def batch_requests():
    """need return_audio=False"""
    # Conversation with video only
    conversation1 = [
        {"role": "system", "content": SPEECH_SYS_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.join(ASSETS_DIR, "draw1.mp4")},
            ],
        },
    ]

    # Conversation with audio only
    conversation2 = [
        {"role": "system", "content": SPEECH_SYS_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        },
    ]

    # Conversation with pure text
    conversation3 = [
        {"role": "system", "content": SPEECH_SYS_PROMPT},
        {"role": "user", "content": "who are you?"},
    ]

    # Conversation with mixed media
    conversation4 = [
        {"role": "system", "content": SPEECH_SYS_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {
                    "type": "text",
                    "text": "What are the elements can you see and hear in these medias?",
                },
            ],
        },
    ]

    # Combine messages for batch processing
    conversations = [conversation1, conversation2, conversation3, conversation4]
    texts, _ = inference(conversations, return_audio=False, use_audio_in_video=True)
    print(texts)


"""
# NOTE: if want to generate speech, need use SPEECH_SYS_PROMPT to generate speech

# asr (audio understanding)
IMAGE_GPU=L4 modal run src/llm/transformers/qwen2_5omni.py --task universal_audio_understanding

# audio to text and speech
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task voice_chatting

# vision(video no audio) to text
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task video_information_extracting
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task screen_recording_interaction

# vision(video with audio) to text and speech
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task omni_chatting_for_math
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task omni_chatting_for_music

# vision(video with audio) to text and speech with multi rounds chat, but need more GPU memory
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen2_5omni.py --task multi_round_omni_chatting

# batch requests
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen2_5omni.py --task batch_requests
"""


@app.local_entrypoint()
def main(task: str = "universal_audio_understanding"):
    tasks = {
        "universal_audio_understanding": universal_audio_understanding,
        "voice_chatting": voice_chatting,
        "video_information_extracting": video_information_extracting,
        "screen_recording_interaction": screen_recording_interaction,
        "omni_chatting_for_math": omni_chatting_for_math,
        "omni_chatting_for_music": omni_chatting_for_music,
        "multi_round_omni_chatting": multi_round_omni_chatting,
        "batch_requests": batch_requests,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
