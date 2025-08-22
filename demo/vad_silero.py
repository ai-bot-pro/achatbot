r"""
- https://github.com/snakers4/silero-vad
- https://medium.com/axinc-ai/silerovad-machine-learning-model-to-detect-speech-segments-e99722c0dd41
"""

import io
import threading

import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import numpy as np
import torch

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

continue_recording = True
# num_samples = 1536
num_samples = 512

audio_file = f"./test/audio_files/asr_example_zh.wav"


def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False


def start_recording():
    from jupyterplot import ProgressPlot

    audio = pyaudio.PyAudio()
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


def record():
    audio = pyaudio.PyAudio()

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


def vad_iter():
    """online vad start and end"""
    vad_iterator = VADIterator(model, sampling_rate=SAMPLE_RATE)
    wav = read_audio(audio_file, sampling_rate=SAMPLE_RATE)
    print(type(wav), wav.shape)

    window_size_samples = 512 if SAMPLE_RATE == 16000 else 256
    for i in range(0, wav.shape[0], window_size_samples):
        chunk = wav[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, window_size_samples - len(chunk)),
                "constant",
                0,
            )
            # print(chunk)
        speech_dict = vad_iterator(chunk, return_seconds=False)
        if speech_dict:
            if "start" in speech_dict:
                speech_dict["start_at"] = round(speech_dict["start"] / SAMPLE_RATE, 3)
            if "end" in speech_dict:
                speech_dict["end_at"] = round(speech_dict["end"] / SAMPLE_RATE, 3)
            print(speech_dict, end=" ")
    print()
    vad_iterator.reset_states()  # reset model states after each audio


def predict():
    """online predict"""
    wav = read_audio(audio_file, sampling_rate=SAMPLE_RATE)
    speech_probs = []
    window_size_samples = 512 if SAMPLE_RATE == 16000 else 256
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, window_size_samples - len(chunk)),
                "constant",
                0,
            )
        speech_prob = model(chunk, SAMPLE_RATE).item()
        speech_probs.append(speech_prob)
    model.reset_states()  # reset model states after each audio

    print(len(speech_probs), speech_probs)


def get_timestamps():
    """offline batch"""
    wav = read_audio(audio_file, sampling_rate=SAMPLE_RATE)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples, cur_samples/sample_rate)
    )
    print(speech_timestamps)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torchaudio.set_audio_backend("soundfile")

    # https://github.com/snakers4/silero-vad/blob/master/hubconf.py
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=True,
    )
    print(utils, model)

    # https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_iter()
    predict()
    get_timestamps()
    # record()
