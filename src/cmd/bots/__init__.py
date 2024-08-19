from . import dummy_bot


def do_register_bots(bot_name: str = "DummyBot"):
    """ import package to register """
    if "Dummy" in bot_name:
        from . import dummy_bot
        return True
    if "DailyRTVI" in bot_name:
        from . import daily_rtvi_bot
        return True
    if "DailyAsrRTVI" in bot_name:
        from . import daily_asr_rtvi_bot
        return True
    if "DailyLangchainRAG" in bot_name:
        from .rag import daily_langchain_rag_bot
        return True
    if "Daily" in bot_name:
        from . import daily_rtvi_bot
        from . import daily_asr_rtvi_bot
        from .rag import daily_langchain_rag_bot
        return True

    return False
