import os
from typing import List, Optional

from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(override=True)


DEFAULT_LLM_SYS_MESSAGES = [
    {
        "role": "system",
        "content": os.getenv("LLM_CHAT_SYSTEM", ""),
    }]

GROQ_LLM_URL = "https://api.groq.com/openai/v1"
GROQ_LLM_MODEL = "llama-3.1-70b-versatile"

# https://docs.together.ai/docs/chat-models
TOGETHER_LLM_URL = "https://api.together.xyz/v1"
TOGETHER_LLM_MODEL = "Qwen/Qwen2-72B-Instruct"

DEFAULT_LLM_URL = os.getenv("LLM_OPENAI_BASE_URL", TOGETHER_LLM_URL)
DEFAULT_LLM_MODEL = os.getenv("LLM_OPENAI_MODEL", TOGETHER_LLM_MODEL)

DEFAULT_LLM_LANG = "zh"


class VADConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class ASRConfig(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class LLMConfig(BaseModel):
    base_url: Optional[str] = DEFAULT_LLM_URL
    model: Optional[str] = DEFAULT_LLM_MODEL
    language: Optional[str] = DEFAULT_LLM_LANG
    messages: Optional[List[dict]] = None
    tag: Optional[str] = "openai_llm_processor"
    args: Optional[dict] = None


class TTSConfig(BaseModel):
    voice: Optional[str] = None
    language: Optional[str] = None
    tag: Optional[str] = None
    args: Optional[dict] = None


class AIConfig(BaseModel):
    vad: Optional[VADConfig] = None
    asr: Optional[ASRConfig] = None
    llm: Optional[LLMConfig] = None
    tts: Optional[TTSConfig] = None
