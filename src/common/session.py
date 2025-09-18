from .types import SessionCtx
from .chat_history import ChatHistory


class Session:
    def __init__(self, **args) -> None:
        chat_history_size = args.pop("chat_history_size", None)
        self.ctx = SessionCtx(**args)
        self.config = {}
        self.chat_round = 0
        self.chat_history = ChatHistory(size=chat_history_size)

    def init_chat_message(self, init_chat_message: dict):
        self.chat_history.init(init_chat_message)

    def reset(self):
        self.chat_round = 0
        self.chat_history.clear()

    def __getstate__(self):
        return {
            "config": self.config,
            "chat_round": self.chat_round,
            "chat_history": self.chat_history,
            "ctx": self.ctx,
        }

    def __setstate__(self, state):
        self.config = state["config"]
        self.chat_round = state["chat_round"]
        self.chat_history = state["chat_history"]
        self.ctx = state["ctx"]

    def __repr__(self) -> str:
        session = {
            "config": self.config,
            "chat_round": self.chat_round,
            "chat_history": self.chat_history,
            "ctx": self.ctx,
        }
        return f"{session}"

    def set_chat_history_size(self, chat_history_size: int | None):
        self.chat_history.set_size(chat_history_size)

    def set_client_id(self, client_id):
        self.ctx.client_id = client_id

    def update_config(self, config_data):
        self.config.update(config_data)

    def append_audio_data(self, audio_data):
        if self.ctx.buffering_strategy is not None:
            self.ctx.buffering_strategy.insert(audio_data)

    def clear_buffer(self):
        if self.ctx.buffering_strategy is not None:
            self.ctx.buffering_strategy.clear()

    def increment_chat_round(self):
        self.chat_round += 1

    def get_record_audio_name(self):
        return f"record_{self.chat_round}_{self.ctx.client_id}.wav"

    def get_paly_audio_name(self):
        return f"play_{self.chat_round}_{self.ctx.client_id}.wav"

    def process_audio(self):
        if self.ctx.on_session_start:
            self.ctx.on_session_start(self)
        if self.ctx.buffering_strategy:
            self.ctx.buffering_strategy.process_audio(self)
        if self.ctx.on_session_end:
            self.ctx.on_session_end(self)
        self.clear_buffer()

    def close(self):
        if hasattr(self.ctx.buffering_strategy, "close"):
            self.ctx.buffering_strategy.close()
        if hasattr(self.ctx.waker, "close"):
            self.ctx.waker.close()
        if hasattr(self.ctx.vad, "close"):
            self.ctx.vad.close()
        if hasattr(self.ctx.asr, "close"):
            self.ctx.asr.close()
        if hasattr(self.ctx.llm, "close"):
            self.ctx.llm.close()
        if hasattr(self.ctx.tts, "close"):
            self.ctx.tts.close()


""""
python -m src.common.session
"""


def test_chat_history():
    chat_history_size = 3
    session = Session(
        chat_history_size=chat_history_size,
        client_id="test_client",
        asr=None,
        llm=None,
        tts=None,
        vad=None,
        waker=None,
        buffering_strategy=None,
        on_session_start=None,
        on_session_end=None,
    )
    print("init", session)

    for i in range(10):
        session.chat_history.append({"role": "user", "content": f"Hello {i}"})
        session.chat_history.append(
            {"role": "assistant", "content": f"Hi, how can I help you {i}?"}
        )
        session.increment_chat_round()
        print(f"{i=} chat", session)
        if i < chat_history_size:
            assert len(session.chat_history.to_list()) == (i + 1) * 2
        else:
            assert len(session.chat_history.to_list()) == chat_history_size * 2

    session.reset()
    print("reset", session)
    assert len(session.chat_history.to_list()) == 0

    print(f"\ntest_chat_history pass\n\n")


def test_pickle():
    import pickle

    chat_history_size = 3
    session = Session(
        chat_history_size=chat_history_size,
        **SessionCtx(client_id="test_client").__dict__,
    )
    print("init", session)

    pickle_data = pickle.dumps(session)
    print("session dump", pickle_data)

    load_session = pickle.loads(pickle_data)
    print("session load", load_session)
    assert str(session) == str(load_session)
    print(f"\ntest_pickle pass\n\n")


"""
python -m src.common.session
"""
if __name__ == "__main__":
    test_pickle()
    test_chat_history()
