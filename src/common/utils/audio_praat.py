"""
frontend
- https://praat.org/
- https://github.com/YannickJadoul/Parselmouth
"""

import warnings

import librosa
import numpy as np
import parselmouth


warnings.filterwarnings("ignore")


def get_energy(chunk, sr, from_harmonic=1, to_harmonic=5):
    sound = parselmouth.Sound(chunk, sampling_frequency=sr)
    # pitch
    # Note: To analyse this Sound, “minimum pitch” must not be less than 280.7017543859649 Hz.
    pitch = sound.to_pitch(pitch_floor=100, pitch_ceiling=350)
    # pitch energy
    # energy = np.mean(pitch.selected_array["strength"])
    pitch = np.mean(pitch.selected_array["frequency"])
    # frame log energy
    # energy = np.mean(sound.to_mfcc().to_array(), axis=1)[0]

    # energy form x-th harmonic to y-th harmonic
    freqs = librosa.fft_frequencies(sr=sr)
    freq_band_idx = np.where((freqs >= from_harmonic * pitch) & (freqs <= to_harmonic * pitch))[0]
    energy = np.sum(np.abs(librosa.stft(chunk)[freq_band_idx, :]))

    return energy


"""
python -m src.common.utils.audio_praat > audio_praat_1.txt
FROM_HARMONIC=2 python -m src.common.utils.audio_praat > audio_praat_2.txt
FROM_HARMONIC=3 python -m src.common.utils.audio_praat > audio_praat_3.txt
FROM_HARMONIC=4 python -m src.common.utils.audio_praat > audio_praat_4.txt

vimdiff audio_praat_1.txt audio_praat_4.txt
"""
if __name__ == "__main__":
    import os

    from src.common.utils.audio_utils import read_wav_to_np

    from_harmonic = int(os.getenv("FROM_HARMONIC", "1"))
    audio_file = "./test/audio_files/asr_example_zh.wav"
    audio_np, sr = read_wav_to_np(audio_file)
    step = int(0.032 * sr)  # seliro vad 16000 sample rate frames
    print(audio_np.shape, sr, step)

    for i in range(0, audio_np.shape[0], step):
        chunk = audio_np[i : i + step]
        if chunk.shape[0] < step:
            padding_size = step - chunk.shape[0]
            chunk = np.pad(chunk, (0, padding_size), "constant")

        energy = get_energy(chunk, sr, from_harmonic=from_harmonic)
        print(energy)
