import multiprocessing
from multiprocessing.synchronize import Event
import struct
import threading
import time
import pyaudio
import wave
import numpy as np

from asr_whisper_faster import faster_whisper_transcribe
from llm_llamacpp import generate


# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000
SILENCE_THRESHOLD = 500
# two seconds of silence marks the end of user voice input
SILENT_CHUNKS = 2 * RATE / CHUNK
# Set microphone id. Use list_microphones.py to see a device list.
MIC_IDX = 1

INT16_MAX_ABS_VALUE = 32768.0


def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i**2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms


audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=MIC_IDX,
    frames_per_buffer=CHUNK,
)


def record_audio(conn):
    silent_chunks = 0
    audio_started = False
    frames = []

    stream.start_stream()
    print("start recording")
    while True:
        data = stream.read(CHUNK)
        rms = compute_rms(data)
        if audio_started:
            frames.append(data)
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            frames.append(data)
            audio_started = True

    stream.stop_stream()

    conn.send(("record frames", frames))
    save_file("records/tmp.wav", audio.get_sample_size(FORMAT), frames)


def save_file(path, sample_width, data: list[bytes]):
    # save audio to a WAV file
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(data))
        print(f"save to {path}")


def loop_record(conn, e: Event):
    while True:
        e.clear()
        record_audio(conn)
        e.wait()


def loop_asr(conn, q: multiprocessing.Queue, download_root, model_size="base", target_lang="zh"):
    while True:
        msg, frames = conn.recv()
        if msg is None or msg == "stop":
            break
        print(f"Received: {msg}")
        # Convert the buffer frames to a NumPy array
        audio_array = np.frombuffer(b"".join(frames), dtype=np.int16)
        # Normalize the array to a [-1, 1] range
        audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        text = faster_whisper_transcribe(audio, download_root, model_size, target_lang)
        if len(text) > 0:
            q.put_nowait(text)


def loop_llm_generate(model_path, q: multiprocessing.Queue, e: Event):
    print(f"loop_llm_generate {model_path} {q}")
    while True:
        if q:
            content = q.get()
            print(f"content {content}")
            generate(model_path, content)
            e.set()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="./models", help="model root path")
    parser.add_argument("--model_size", "-s", type=str, default="base", help="model size")
    parser.add_argument("--lang", "-l", type=str, default="zh", help="target language")
    parser.add_argument(
        "--llm_model_path",
        "-lm",
        type=str,
        default="./models/qwen2-1_5b-instruct-q8_0.gguf",
        help="llm model path",
    )
    args = parser.parse_args()

    mp_queue = multiprocessing.Queue()
    start_record_event = multiprocessing.Event()
    parent_conn, child_conn = multiprocessing.Pipe()

    # p = multiprocessing.Process(target=record_audio, args=(child_conn,))
    p = multiprocessing.Process(target=loop_record, args=(child_conn, start_record_event))
    c = multiprocessing.Process(
        target=loop_asr, args=(parent_conn, mp_queue, args.model_path, args.model_size, args.lang)
    )
    t = threading.Thread(
        target=loop_llm_generate, args=(args.llm_model_path, mp_queue, start_record_event)
    )
    p.start()
    c.start()
    t.start()

    t.join()
    p.join()
    c.join()

    start_record_event.clear()
    mp_queue.close()

    stream.close()
    audio.terminate()
