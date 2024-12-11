import logging

import pyaudio

from src.common.interface import IAudioStream
from src.common.factory import EngineClass
from src.types.speech.audio_stream import AudioStreamInfo, PyAudioStreamArgs


class PyAudioStream(EngineClass, IAudioStream):
    TAG = [
        "pyaudio_stream",
        "pyaudio_in_stream",
        "pyaudio_out_stream",
    ]

    def __init__(self, **args):
        self.args = PyAudioStreamArgs(**args)
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        if self.args.input is True and self.args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_input_device_info()
            self.args.input_device_index = default_device["index"]
            # self.args.rate = int(default_device['defaultSampleRate'])
        if self.args.output is True and self.args.input_device_index is None:
            default_device = self.pyaudio_instance.get_default_output_device_info()
            self.args.output_device_index = default_device["index"]

        logging.info(f"PyAudioStreamArgs: {self.args}")

    def open_stream(self):
        """Opens an audio stream."""
        if self.stream:
            return

        pyChannels = self.args.channels
        pySampleRate = self.args.rate
        pyFormat = self.args.format
        # check for mpeg format
        if pyFormat == pyaudio.paCustomFormat:
            pyFormat = self.pyaudio_instance.get_format_from_width(2)
        logging.debug(
            "Opening stream for wave audio chunks, "
            f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
            f"pySampleRate: {pySampleRate}"
        )
        try:
            self.stream = self.pyaudio_instance.open(
                format=int(pyFormat),
                channels=pyChannels,
                rate=pySampleRate,
                output_device_index=self.args.output_device_index,
                input_device_index=self.args.input_device_index,
                output=self.args.output,
                input=self.args.input,
                frames_per_buffer=self.args.frames_per_buffer,
                stream_callback=self.args.stream_callback,
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
        if self.stream is None:
            return False

        return self.stream and self.stream.is_active()

    def write_stream(self, data):
        self.stream.write(data)

    def get_stream_info(self) -> AudioStreamInfo:
        return AudioStreamInfo(
            in_channels=self.args.channels,
            in_sample_rate=self.args.rate,
            in_sample_width=self.args.sample_width,
            in_frames_per_buffer=self.args.frames_per_buffer,
            out_channels=self.args.channels,
            out_sample_rate=self.args.rate,
            out_sample_width=self.args.sample_width,
            out_frames_per_buffer=self.args.frames_per_buffer,
            pyaudio_out_format=self.args.format,
        )

    def read_stream(self, num_frames):
        return self.stream.read(num_frames)

    def close(self) -> None:
        """Closes the audio instance."""
        if self.stream:
            self.close_stream()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
