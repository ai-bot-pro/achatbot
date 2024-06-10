from .types import SessionCtx


class Session:
    def __init__(self, **args: SessionCtx) -> None:
        self.ctx = SessionCtx(**args)
        self.config = {}
        self.file_counter = 0

    def update_config(self, config_data):
        self.config.update(config_data)

    def append_audio_data(self, audio_data):
        self.ctx.buffering_stragegy.insert(audio_data)

    def clear_buffer(self):
        self.ctx.buffering_stragegy.clear()

    def increment_file_counter(self):
        self.file_counter += 1

    def get_file_name(self):
        return f"{self.ctx.client_id}_{self.file_counter}.wav"

    def process_audio(self):
        # @TODO: use burr work flow
        if self.ctx.on_session_start is not None:
            self.ctx.on_session_start(self)
        self.ctx.buffering_stragegy.process_audio(self)
        if self.ctx.on_session_end is not None:
            self.ctx.on_session_end(self)
        self.clear_buffer()
