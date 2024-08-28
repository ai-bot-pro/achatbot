from src.common.register import Register

register_daily_room_bots = Register('daily-room-bots')


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
    if "Daily" in bot_name:
        from .rtvi import daily_rtvi_bot
        from .rtvi import daily_asr_rtvi_bot
        from .rag import daily_langchain_rag_bot
        return True

    return False
