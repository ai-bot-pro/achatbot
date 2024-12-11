import itertools
import os
import collections
import struct

import pyaudio
import pvporcupine
import wave


porcupine = None
paud = None
audio_stream = None

access_key = os.environ.get("PORCUPINE_ACCESS_KEY")


def get_defualt_instance():
    keywords = "hey google,terminator,americano,computer,pico clock,hey siri,grasshopper,alexa,jarvis,bumblebee,hey barista,blueberry,ok google,grapefruit,picovoice,porcupine".split(
        ","
    )
    print(keywords)
    porcupine = pvporcupine.create(access_key, keywords=keywords)
    return porcupine, keywords


def get_custom_instance():
    keywords = "小黑".split(",")
    porcupine = pvporcupine.create(
        access_key,
        keywords=keywords,
        model_path="./models/porcupine_params_zh.pv",
        keyword_paths=["./models/小黑_zh_mac_v3_0_0.ppn"],
    )
    return porcupine, keywords


def porcess_wakeword(porcupine, audio_buffer):
    # Removing the wake word from the recording
    samples_for_0_1_sec = int(porcupine.sample_rate * 0.1)
    start_index = max(0, len(audio_buffer) - samples_for_0_1_sec)
    print(f"start_index {start_index}, samples_for_0_1_sec {samples_for_0_1_sec}")
    temp_samples = collections.deque(itertools.islice(audio_buffer, start_index, None))
    audio_buffer.clear()
    audio_buffer.extend(temp_samples)
    return audio_buffer


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def export_wav(data, filename, sample_rate=RATE, channels=CHANNELS, sample_width=2):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(data))
    wf.close()


try:
    # porcupine = get_defualt_instance()
    porcupine, keywords = get_custom_instance()
    print(porcupine.sample_rate, porcupine.frame_length)
    if porcupine.sample_rate != RATE:
        raise ValueError("Sample rate of your device is not %d Hz" % RATE)

    pre_recording_buffer_duration = 10.0
    maxlen = int((porcupine.sample_rate // porcupine.frame_length) * pre_recording_buffer_duration)
    print(f"audio_buffer maxlen: {maxlen}")
    # ring buffer
    audio_buffer = collections.deque(maxlen=maxlen)

    paud = pyaudio.PyAudio()
    audio_stream = paud.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )
    print("start recording")
    while True:
        read_audio_frames = audio_stream.read(porcupine.frame_length)
        # print(type(keyword), len(keyword))
        keyword = struct.unpack_from("h" * porcupine.frame_length, read_audio_frames)
        # print(type(keyword), len(keyword))
        keyword_index = porcupine.process(keyword)
        if keyword_index >= 0:
            print(
                f"index {keyword_index} {keywords[keyword_index]} hotword detected, audio_buffer length {len(audio_buffer)}"
            )

            audio_buffer = porcess_wakeword(porcupine, audio_buffer)
            print(f"audio_buffer length {len(audio_buffer)}")
            export_wav(
                audio_buffer,
                "./records/tmp_wakeword_porcupine.wav",
                sample_rate=porcupine.sample_rate,
                channels=CHANNELS,
                sample_width=paud.get_sample_size(FORMAT),
            )

        audio_buffer.append(read_audio_frames)
        # print(len(audio_buffer))

finally:
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if paud is not None:
        paud.terminate()
