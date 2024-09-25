import collections
import queue


class RingBuffer(object):
    """Ring buffer to hold audio from audio stream"""

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)

    def get_buf(self):
        return self._buf

    def lenght(self):
        return len(self._buf)

    def append(self, data, at: str = "end"):
        """Add a data to the front/end of buffer"""
        match at:
            case "front":
                self._buf.appendleft(data)
            case "end":
                self._buf.append(data)
            case _:
                self._buf.append(data)

    def extend(self, data, at: str = "end"):
        """Adds data to the front/end of buffer"""
        match at:
            case "front":
                self._buf.extendleft(data)
            case "end":
                self._buf.extend(data)
            case _:
                self._buf.extend(data)

    def get(self, is_clear=True, cls=bytes):
        """Retrieves data from the beginning of buffer and clears it"""
        if issubclass(cls, bytes):
            tmp = bytes(bytearray(self._buf))
        else:
            tmp = list(self._buf)
        is_clear and self._buf.clear()
        return tmp


class AudioBufferManager:
    """
    Manages an audio buffer, allowing addition and retrieval of audio data.
    """

    def __init__(self, audio_buffer: queue.Queue):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
        """
        self.audio_buffer = audio_buffer
        self.total_samples = 0

    def add_to_buffer(self, audio_data):
        """
        Adds audio data to the buffer.

        Args:
            audio_data: Audio data to be added.
        """
        self.audio_buffer.put(audio_data)
        self.total_samples += len(audio_data) // 2

    def empty(self):
        return self.audio_buffer.empty()

    def clear_buffer(self):
        """Clears all audio data from the buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                continue
        self.total_samples = 0

    def get_from_buffer(self, timeout: float = 0.05):
        """
        Retrieves audio data from the buffer.

        Args:
            timeout (float): Time (in seconds) to wait
              before raising a queue.Empty exception.

        Returns:
            The audio data chunk or None if the buffer is empty.
        """
        try:
            chunk = self.audio_buffer.get(timeout=timeout)
            self.total_samples -= len(chunk) // 2
            return chunk
        except queue.Empty:
            return None

    def get_buffered_seconds(self, rate: int) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Args:
            rate (int): Sample rate of the audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        return self.total_samples / rate
