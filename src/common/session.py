from .types import SessionCtx


class Session:
    def __init__(self, **args) -> None:
        self.ctx = SessionCtx(**args)
        self.config = {}
        self.chat_round = 0
        # just for local history,@todo: use kv store history like mem0
        self.chat_history = []

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
        d = {
            "config": self.config,
            "chat_round": self.chat_round,
            "chat_history": self.chat_history,
            "ctx": self.ctx,
        }
        s = f"session: {d}"
        return s

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
