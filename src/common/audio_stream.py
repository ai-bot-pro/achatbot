import logging
import collections

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


class AudioStream:
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    def __init__(self, args: AudioStreamArgs):
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        if args.input is True and args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_input_device_info()
            args.input_device_index = default_device['index']
        if args.output is True and args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_output_device_info()
            args.output_device_index = default_device['index']

        logging.info(f"AudioStreamArgs: {args}")
        self.args = args

    def open_stream(self):
        """Opens an audio stream."""

        # check for mpeg format
        pyChannels = self.args.channels
        pySampleRate = self.args.rate
        pyFormat = self.args.format_
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
        except Exception as e:
            raise Exception(f"Error opening stream: {e}")

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
            self.pyaudio_instance.terminate()
