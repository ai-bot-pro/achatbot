from pydantic import BaseModel

from src.common.register import Register

register_daily_room_bots = Register('daily-room-bots')


class BotInfo(BaseModel):
    is_agent: bool = False
    chat_bot_name: str = ""
    config: dict = {}  # @deprecated use config_list options to conf
    room_name: str = "chat-room"
    room_url: str = ""
    token: str = ""
    config_list: list = []
    services: dict = {}
    bot_type: str = "daily"  # default: daily webrtc


def import_bots(bot_name: str = "DummyBot"):
    """ import package to register """
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

    return False
