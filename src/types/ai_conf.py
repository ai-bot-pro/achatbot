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


class StreamConfig(BaseModel):
    tag: Optional[str] = "daily_room_stream"
    args: Optional[dict] = None


class VADConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class VisionDetectorConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class VisionOCRConfig(BaseModel):
    trigger_texts: Optional[List[str]] = None
    tag: Optional[str] = None
    args: Optional[dict] = None


class ImageGenConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class ASRConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class AvatarConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class LLMConfig(BaseModel):
    base_url: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    messages: Optional[List[dict]] = None
    tools: Optional[List[dict]] = None
    # is_use_tools_description: Optional[bool] = False
    tag: Optional[str] = None
    args: Optional[dict] = None


class TTSConfig(BaseModel):
    voice: Optional[str] = None
    language: Optional[str] = None
    tag: Optional[str] = None
    args: Optional[dict] = None


class MCPServerConfig(BaseModel):
    transport: Optional[str] = "stdio"
    parameters: Optional[Dict[str, Any]] = None


class AIConfig(BaseModel):
    vad: Optional[VADConfig] = None
    asr: Optional[ASRConfig] = None
    avatar: Optional[AvatarConfig] = None
    mcp_servers: Optional[Dict[str, MCPServerConfig]] = None
    llm: Optional[LLMConfig] = None
    nlp_task_llm: Optional[LLMConfig] = None
    voice_llm: Optional[LLMConfig] = None
    vision_llm: Optional[LLMConfig] = None
    omni_llm: Optional[LLMConfig] = None
    vision_detector: Optional[VisionDetectorConfig] = None
    vision_ocr: Optional[VisionOCRConfig] = None
    tts: Optional[TTSConfig] = None
    img_gen: Optional[ImageGenConfig] = None
    # TODO: @weedge
    # - use local pyaudio/cv2 streaming;
    # - use remote RTC livekit or agora streaming
    # need to add stream config
    # stream: Optional[StreamConfig] = None
    extends: Optional[dict] = None
