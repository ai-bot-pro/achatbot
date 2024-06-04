from common.types import SessionCtx


class Session:
    def __init__(self, args: SessionCtx) -> None:
        self.client_id = args.client_id
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        self.config = {}
        self.file_counter = 0
        self.args = args

    def update_config(self, config_data):
        self.config.update(config_data)

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)

    def clear_buffer(self):
        self.buffer.clear()

    def increment_file_counter(self):
        self.file_counter += 1

    def get_file_name(self):
        return f"{self.client_id}_{self.file_counter}.wav"
