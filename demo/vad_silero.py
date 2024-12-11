r"""
- https://github.com/snakers4/silero-vad
- https://medium.com/axinc-ai/silerovad-machine-learning-model-to-detect-speech-segments-e99722c0dd41
"""

import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import io
import numpy as np
import torch

from jupyterplot import ProgressPlot
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

continue_recording = True


def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False


def start_recording():
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK
    )

    data = []
    voiced_confidences = []

    global continue_recording
    continue_recording = True

    pp = ProgressPlot(
        plot_names=["Silero VAD"], line_names=["speech probabilities"], x_label="audio chunks"
    )

    stop_listener = threading.Thread(target=stop)
    stop_listener.start()

    while continue_recording:
        audio_chunk = stream.read(num_samples)

        # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)

        pp.update(new_confidence)

    pp.finalize()


# Taken from utils_vad.py
def validate(model, inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


if __name__ == "__main__":
    torch.set_num_threads(1)
    torchaudio.set_audio_backend("soundfile")

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    print(utils, model)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    audio = pyaudio.PyAudio()

    # num_samples = 1536
    num_samples = 512
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK
    )
    data = []
    voiced_confidences = []
    frames_to_record = 100

    print("Started Recording")
    for i in range(0, frames_to_record):
        audio_chunk = stream.read(num_samples)

        # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)
        print(len(audio_float32))

        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)

    print("Stopped the recording")

    # plot the confidences for the speech
    plt.figure(figsize=(20, 6))
    plt.plot(voiced_confidences)
    plt.show()
