from . import dummy_bot


def import_bots(bot_name: str = "DummyBot"):
    """ import package to register """
    if "DummyBot" in bot_name:
        from . import dummy_bot
        return True
    if "DailyRTVIBot" in bot_name:
        from . import daily_rtvi_bot
        return True
    if "DailyAsrRTVIBot" in bot_name:
        from . import daily_asr_rtvi_bot
        return True
    if "DailyBot" in bot_name:
        from . import daily_bot
        return True
    if "DailyLangchainRAGBot" in bot_name:
        from .rag import daily_langchain_rag_bot
        return True
    if "Daily" in bot_name:
        from . import daily_rtvi_bot
        from . import daily_asr_rtvi_bot
        from .rag import daily_langchain_rag_bot
        return True

    return False
