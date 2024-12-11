from dataclasses import dataclass

import deps.ChatTTS.ChatTTS as ChatTTS


@dataclass
class ChatTTSArgs:
    source: str = "huggingface"  # "huggingface", "local", "custom"
    force_redownload: bool = False
    local_path: str = ""
    compile: bool = True
    device: str = None
    use_flash_attn: bool = False
    skip_refine_text: bool = False
    refine_text_only: bool = False
    params_refine_text: ChatTTS.Chat.RefineTextParams | None = None
    params_infer_code: ChatTTS.Chat.InferCodeParams | None = None
    use_decoder: bool = True
    do_text_normalization: bool = False
    lang: str = None
    tts_stream: bool = False
