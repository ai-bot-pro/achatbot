import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv(override=True)


DEFAULT_LLM_SYS_MESSAGES = [
    {
        "role": "system",
        "content": os.getenv("LLM_CHAT_SYSTEM", ""),
    }
]

# https://console.groq.com/docs/models
GROQ_LLM_URL = "https://api.groq.com/openai/v1"
GROQ_LLM_MODEL = "llama-3.1-70b-versatile"

# https://docs.together.ai/docs/chat-models
TOGETHER_LLM_URL = "https://api.together.xyz/v1"
TOGETHER_LLM_MODEL = "Qwen/Qwen2-72B-Instruct"

DEFAULT_LLM_URL = os.getenv("LLM_OPENAI_BASE_URL", TOGETHER_LLM_URL)
DEFAULT_LLM_MODEL = os.getenv("LLM_OPENAI_MODEL", TOGETHER_LLM_MODEL)

DEFAULT_LLM_LANG = "zh"


class BaseConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None
    pool_size: Optional[int] = None
    pool_init_worker_num: Optional[int] = None


class MCPServerConfig(BaseModel):
    transport: Optional[str] = "stdio"
    parameters: Optional[Dict[str, Any]] = None


class StreamConfig(BaseModel):
    tag: Optional[str] = "daily_room_stream"


class VADConfig(BaseConfig):
    pass


class SEConfig(BaseConfig):
    pass


class TurnConfig(BaseConfig):
    pass


class VisionDetectorConfig(BaseConfig):
    pass


class VisionOCRConfig(BaseConfig):
    trigger_texts: Optional[List[str]] = None


class ImageGenConfig(BaseConfig):
    pass


class ASRConfig(BaseConfig):
    pass


class PuncConfig(BaseConfig):
    pass


class AvatarConfig(BaseConfig):
    pass


class MemoryConfig(BaseConfig):
    processor: Optional[str] = None


class LLMConfig(BaseConfig):
    init_prompt: Optional[str] = None
    processor: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    messages: Optional[List[dict]] = None
    tools: Optional[List[dict]] = None
    # is_use_tools_description: Optional[bool] = False


class TranslateLLMConfig(BaseConfig):
    init_prompt: Optional[str] = None
    model: Optional[str] = None
    src: Optional[str] = None
    target: Optional[str] = None
    streaming: Optional[bool] = False
    prompt_tpl: Optional[str] = None


class TTSConfig(BaseConfig):
    voice: Optional[str] = None
    language: Optional[str] = None
    aggregate_sentences: Optional[bool] = True
    push_text_frames: Optional[bool] = True
    remove_punctuation: Optional[bool] = False


class A2AConfig(BaseConfig):
    language: Optional[str] = None


class AIConfig(BaseModel):
    vad: Optional[VADConfig] = None
    se: Optional[SEConfig] = None
    turn: Optional[TurnConfig] = None
    asr: Optional[ASRConfig] = None
    punctuation: Optional[PuncConfig] = None
    avatar: Optional[AvatarConfig] = None
    a2a: Optional[A2AConfig] = None
    mcp_servers: Optional[Dict[str, MCPServerConfig]] = None
    llm: Optional[LLMConfig] = None
    translate_llm: Optional[TranslateLLMConfig] = None
    nlp_task_llm: Optional[LLMConfig] = None
    voice_llm: Optional[LLMConfig] = None
    vision_llm: Optional[LLMConfig] = None
    omni_llm: Optional[LLMConfig] = None
    vision_detector: Optional[VisionDetectorConfig] = None
    vision_ocr: Optional[VisionOCRConfig] = None
    tts: Optional[TTSConfig] = None
    img_gen: Optional[ImageGenConfig] = None
    memory: Optional[MemoryConfig] = None
    # TODO: @weedge
    # - use local pyaudio/cv2 streaming;
    # - use remote RTC livekit or agora streaming
    # need to add stream config
    # stream: Optional[StreamConfig] = None
    extends: Optional[dict] = None
