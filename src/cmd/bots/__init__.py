from pydantic import BaseModel

from src.common.register import Register
from src.common.const import *

register_ai_room_bots = Register("ai-room-bots")
register_ai_fastapi_ws_bots = Register("fastapi-ws-bots")
register_ai_small_webrtc_bots = Register("small-webrtc-bots")


class BotInfo(BaseModel):
    is_agent: bool = False
    is_background: bool = True  # background task
    chat_bot_name: str = ""
    config: dict = {}  # @deprecated use config_list options to conf
    room_name: str = "chat-room"
    room_url: str = ""
    token: str = ""
    room_expire: int = ROOM_EXPIRE_TIME
    config_list: list = []
    services: dict = {}
    websocket_server_host: str = "localhost"
    websocket_server_port: int = 8765
    transport_type: str = "room"  # room(daily,livekit), websocket(websocket,fastapi_websocket)
    handle_sigint: bool = True


def import_bots(bot_name: str = "DummyBot"):
    """import package to register"""
    if "DummyBot" in bot_name:
        from . import dummy_bot

        return True
    if "DailyRTVIBot" in bot_name:
        from .rtvi import daily_rtvi_bot

        return True
    if "DailyAsrRTVIBot" in bot_name:
        from .rtvi import daily_asr_rtvi_bot

        return True
    if "DailyBot" in bot_name:
        from . import daily_bot

        return True
    if "DailyLangchainRAGBot" in bot_name:
        from .rag import daily_langchain_rag_bot

        return True
    if "DailyRTVIGeneralBot" in bot_name:
        from .rtvi import daily_rtvi_general_bot

        return True
    if "DailyEchoVisionBot" in bot_name:
        from .vision import daily_echo_vision_bot

        return True
    if "DailyDescribeVisionBot" in bot_name:
        from .vision import daily_describe_vision_bot

        return True
    if "DailyMockVisionBot" in bot_name:
        from .vision import daily_mock_vision_bot

        return True
    if "DailyChatVisionBot" in bot_name:
        from .vision import daily_chat_vision_bot

        return True
    if "DailyChatToolsVisionBot" in bot_name:
        from .vision import daily_chat_tools_vision_bot

        return True
    if "DailyAnnotateVisionBot" in bot_name:
        from .vision import daily_annotate_vision_bot

        return True
    if "DailyDetectVisionBot" in bot_name:
        from .vision import daily_detect_vision_bot

        return True
    if "DailyOCRVisionBot" in bot_name:
        from .vision import daily_ocr_vision_bot

        return True
    if "DailyDescribeVisionToolsBot" in bot_name:
        from .vision import daily_describe_vision_tools_bot

        return True
    if "DailyMonthNarrationBot" in bot_name:
        from .image import daily_month_narration_bot

        return True
    if "DailyStoryTellingBot" in bot_name:
        from .image.storytelling import daily_bot

        return True
    if "DailyNaturalConversationBot" in bot_name:
        from .nlp import daily_natural_conversation_bot

        return True
    if "DailyGLMVoiceBot" in bot_name:
        from .voice import daily_glm_voice_bot

        return True
    if "DailyAsrGLMVoiceBot" in bot_name:
        from .voice import daily_asr_glm_voice_bot

        return True
    if "DailyFreezeOmniVoiceBot" in bot_name:
        from .voice import daily_freeze_omni_voice_bot

        return True
    if "DailyMiniCPMoVoiceBot" in bot_name:
        from .voice import daily_minicpmo_voice_bot

        return True
    if "DailyAsrMiniCPMoVoiceBot" in bot_name:
        from .voice import daily_asr_minicpmo_voice_bot

        return True
    if "DailyMiniCPMoVisionVoiceBot" in bot_name:
        from .omni import daily_minicpmo_vision_voice_bot

        return True
    if "DailyStepVoiceBot" in bot_name:
        from .voice import daily_step_voice_bot

        return True
    if "DailyAsrStepVoiceBot" in bot_name:
        from .voice import daily_asr_step_voice_bot

        return True
    if "DailyPhi4VisionSpeechBot" in bot_name:
        from .omni import daily_phi4_vision_speech_bot

        return True
    if "DailyPhi4VoiceBot" in bot_name:
        from .voice import daily_phi4_voice_bot

        return True
    if "DailyNASABot" in bot_name:
        from .mcp import daily_nasa_bot

        return True
    if "DailyMultiMCPBot" in bot_name:
        from .mcp import daily_multi_mcp_bot

        return True
    if "DailyAvatarEchoBot" in bot_name:
        from .avatar import daily_liteavatar_echo_bot

        return True
    if "DailyAvatarChatBot" in bot_name:
        from .avatar import daily_liteavatar_chat_bot

        return True
    if "LivekitBot" in bot_name:
        from . import livekit_bot

        return True
    if "LivekitDescribeVisionBot" in bot_name:
        from .vision import livekit_describe_vision_bot

        return True
    if "LivekitEchoVisionBot" in bot_name:
        from .vision import livekit_echo_vision_bot

        return True
    if "LivekitMockVisionBot" in bot_name:
        from .vision import livekit_mock_vision_bot

        return True
    if "LivekitChatVisionBot" in bot_name:
        from .vision import livekit_chat_vision_bot

        return True
    if "LivekitChatToolsVisionBot" in bot_name:
        from .vision import livekit_chat_tools_vision_bot

        return True
    if "LivekitAnnotateVisionBot" in bot_name:
        from .vision import livekit_annotate_vision_bot

        return True
    if "LivekitDetectVisionBot" in bot_name:
        from .vision import livekit_detect_vision_bot

        return True
    if "LivekitOCRVisionBot" in bot_name:
        from .vision import livekit_ocr_vision_bot

        return True
    if "LivekitDescribeVisionToolsBot" in bot_name:
        from .vision import livekit_describe_vision_tools_bot

        return True
    if "LivekitQwen2_5OmniVoiceBot" in bot_name:
        from .voice import livekit_qwen2_5omni_voice_bot

        return True
    if "LivekitAsrQwen2_5OmniVoiceBot" in bot_name:
        from .voice import livekit_asr_qwen2_5omni_voice_bot

        return True
    if "LivekitQwen2_5OmniVisionVoiceBot" in bot_name:
        from .omni import livekit_qwen2_5omni_vision_voice_bot

        return True
    if "LivekitAsrKimiVoiceBot" in bot_name:
        from .voice import livekit_asr_kimi_voice_bot

        return True
    if "LivekitKimiVoiceBot" in bot_name:
        from .voice import livekit_kimi_voice_bot

        return True
    if "LivekitAsrVITAVoiceBot" in bot_name:
        from .voice import livekit_asr_vita_voice_bot

        return True
    if "LivekitVITAVoiceBot" in bot_name:
        from .voice import livekit_vita_voice_bot

        return True
    if "LivekitNASABot" in bot_name:
        from .mcp import livekit_nasa_bot

        return True
    if "LivekitMultiMCPBot" in bot_name:
        from .mcp import livekit_multi_mcp_bot

        return True
    if "LivekitAvatarEchoBot" in bot_name:
        from .avatar import livekit_musetalk_echo_bot

        return True
    if "LivekitAvatarChatBot" in bot_name:
        from .avatar import livekit_musetalk_chat_bot

        return True
    # if "LivekitMoshiVoiceBot" in bot_name:
    #    from .voice import livekit_moshi_bot

    #    return True
    if "AgoraBot" in bot_name:
        from . import agora_bot

        return True
    if "AgoraEchoVisionBot" in bot_name:
        from .vision import agora_echo_vision_bot

        return True
    if "AgoraMockVisionBot" in bot_name:
        from .vision import agora_mock_vision_bot

        return True
    if "AgoraAnnotateVisionBot" in bot_name:
        from .vision import agora_annotate_vision_bot

        return True
    if "AgoraChatToolsVisionBot" in bot_name:
        from .vision import agora_chat_tools_vision_bot

        return True
    if "AgoraChatVisionBot" in bot_name:
        from .vision import agora_chat_vision_bot

        return True
    if "AgoraDescribeVisionBot" in bot_name:
        from .vision import agora_describe_vision_bot

        return True
    if "AgoraDescribeVisionToolsBot" in bot_name:
        from .vision import agora_describe_vision_tools_bot

        return True
    if "AgoraDetectVisionBot" in bot_name:
        from .vision import agora_detect_vision_bot

        return True
    if "AgoraOCRVisionBot" in bot_name:
        from .vision import agora_ocr_vision_bot

        return True
    if "AgoraNASABot" in bot_name:
        from .mcp import agora_nasa_bot

        return True
    if "AgoraMultiMCPBot" in bot_name:
        from .mcp import agora_multi_mcp_bot

        return True

    return False


def import_websocket_bots(bot_name: str = "DummyBot"):
    if "WebsocketServerBot" in bot_name:
        from . import websocket_server_bot

        return True

    return False


def import_fastapi_websocket_bots(bot_name: str = "DummyBot"):
    if "FastapiWebsocketServerBot" in bot_name:
        from . import fastapi_websocket_server_bot

        return True

    if "FastapiWebsocketMoshiVoiceBot" in bot_name:
        from .voice import fastapi_websocket_moshi_bot

        return True

    if "SmallWebRTCFastapiWebsocketEchoBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_echo_bot

        return True

    if "SmallWebRTCFastapiWebsocketAvatarEchoBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_avatar_echo_bot

        return True

    if "SmallWebRTCFastapiWebsocketAvatarChatBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_avatar_chat_bot

        return True

    return False


def import_small_webrtc_bots(bot_name: str = "DummyBot"):
    if "SmallWebrtcBot" == bot_name:
        from . import small_webrtc_bot

        return True

    if "SmallWebRTCFastapiWebsocketEchoBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_echo_bot

        return True

    if "SmallWebRTCFastapiWebsocketAvatarEchoBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_avatar_echo_bot

        return True

    if "SmallWebRTCFastapiWebsocketAvatarChatBot" in bot_name:
        from .bridge import small_webrtc_fastapi_websocket_avatar_chat_bot

        return True

    return False
