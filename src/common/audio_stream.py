import logging
import collections
import queue

import pyaudio

from src.common.types import AudioStreamArgs


class RingBuffer(object):
    """Ring buffer to hold audio from audio stream"""

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)

    def get_buf(self):
        return self._buf

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the beginning of buffer and clears it"""
        tmp = bytes(bytearray(self._buf))
        self._buf.clear()
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


class AudioStream:
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    def __init__(self, args: AudioStreamArgs):
        self.args = args
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        if args.input is True and args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_input_device_info()
            self.args.input_device_index = default_device['index']
            self.args.rate = int(default_device['defaultSampleRate'])
        if args.output is True and args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_output_device_info()
            args.output_device_index = default_device['index']

        logging.info(f"AudioStreamArgs: {self.args}")

    def open_stream(self):
        """Opens an audio stream."""
        if self.stream:
            return

        pyChannels = self.args.channels
        pySampleRate = self.args.rate
        pyFormat = self.args.format_
        # check for mpeg format
        if pyFormat == pyaudio.paCustomFormat:
            pyFormat = self.pyaudio_instance.get_format_from_width(2)
        logging.debug("Opening stream for wave audio chunks, "
                      f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                      f"pySampleRate: {pySampleRate}")
        try:
            self.stream = self.pyaudio_instance.open(
                format=pyFormat,
                channels=pyChannels,
                rate=pySampleRate,
                output_device_index=self.args.output_device_index,
                input_device_index=self.args.input_device_index,
                output=self.args.output,
                input=self.args.input,
                frames_per_buffer=self.args.frames_per_buffer
            )
        except Exception as ex:
            raise Exception(f"Error opening stream: {ex}")

    def start_stream(self):
        """Starts the audio stream."""
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def close_stream(self):
        """Closes the audio stream."""
        if self.stream:
            self.stop_stream()
            self.stream.close()
            self.stream = None

    def is_stream_active(self) -> bool:
        """
        Checks if the audio stream is active.

        Returns:
            bool: True if the stream is active, False otherwise.
        """
        return self.stream and self.stream.is_active()

    def close(self) -> None:
        """Closes the audio instance."""
        if self.stream:
            self.close_stream()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
