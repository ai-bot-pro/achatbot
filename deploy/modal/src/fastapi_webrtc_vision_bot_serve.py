import modal
import os


class ContainerRuntimeConfig:
    images = {
        # image key name is shot, lenght less 10
        "default": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.3",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
        "qwen": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "llm_transformers_manual_vision_qwen,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.3",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "Qwen/Qwen2-VL-2B-Instruct")}',
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
        "llama": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "llm_transformers_manual_vision_llama,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.3",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "unsloth/Llama-3.2-11B-Vision-Instruct")}',
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
        "janus": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "llm_transformers_manual_vision_img_janus,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.7",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "deepseek-ai/Janus-Pro-1B")}',
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
        "deepseekvl2": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "llm_transformers_manual_vision_deepseekvl2,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.9",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/",
                # extra_index_url="https://test.pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "deepseek-ai/deepseek-vl2-tiny")}',
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
        "minicpmo": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "llm_transformers_manual_vision_voice_minicpmo,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.12",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
                # extra_index_url="https://test.pypi.org/simple/",
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    # llm processor model, default:google gemini_flash_latest
                    "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "openbmb/MiniCPM-o-2_6")}',
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
    }

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]

    @staticmethod
    def get_app_name(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        app_name = "fastapi_webrtc_vision_bot"
        if image_name != "default":
            app_name = f"fastapi_webrtc_vision_{image_name}_bot"
        print(f"app_name:{app_name}")
        return app_name

    @staticmethod
    def get_gpu():
        # https://modal.com/docs/reference/modal.gpu
        # T4, L4, A10G, L40S, A100, A100-80GB, H100
        gpu = os.getenv("IMAGE_GPU", None)
        print(f"image_gpu:{gpu}")
        return gpu

    @staticmethod
    def get_allow_concurrent_inputs():
        concurrent_cn = int(os.getenv("IMAGE_CONCURRENT_CN", "1"))
        print(f"image_concurrent_cn:{concurrent_cn}")
        return concurrent_cn


# ----------------------- app -------------------------------
app = modal.App(ContainerRuntimeConfig.get_app_name())


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    gpu=ContainerRuntimeConfig.get_gpu(),
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    scaledown_window=300,
    timeout=600,
    allow_concurrent_inputs=ContainerRuntimeConfig.get_allow_concurrent_inputs(),
)
class Srv:
    @modal.build()
    def download_model(self):
        # https://huggingface.co/docs/huggingface_hub/guides/download
        from huggingface_hub import snapshot_download
        from achatbot.common.types import MODELS_DIR

        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"start downloading model to dir:{MODELS_DIR}")

        # asr model repo
        if "sense_voice_asr" in os.getenv("ASR_TAG", "sense_voice_asr"):
            snapshot_download(
                repo_id="FunAudioLLM/SenseVoiceSmall",
                repo_type="model",
                allow_patterns="*",
                local_dir=os.path.join(MODELS_DIR, "FunAudioLLM/SenseVoiceSmall"),
            )

        # llm model repo
        llm_model_repo = os.getenv("LLM_MODEL_NAME_OR_PATH")
        if llm_model_repo:
            repo_id = "/".join(llm_model_repo.split("/")[-2:])
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*",
                local_dir=os.path.join(MODELS_DIR, repo_id),
            )

        print("download model done")

    @modal.enter()
    def enter(self):
        print("start enter")

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


if __name__ == "__main__":
    pass
