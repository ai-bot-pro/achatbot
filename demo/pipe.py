import multiprocessing
import struct
import pyaudio
import wave
from collections import deque
import asyncio
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
# Set microphone id. Use tools/list_microphones.py to see a device list.
MIC_IDX = 0

INT16_MAX_ABS_VALUE = 32768.0


g_text_dqueue = deque(maxlen=100)


def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms


def record_audio(conn,):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    print("start recording")
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    conn.send(("record frames", frames))
    save_file('records/tmp.wav',
              audio.get_sample_size(FORMAT), frames)

    content = conn.recv()
    print(f"recv {content}")


def save_file(path, sample_width, data):
    # save audio to a WAV file
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(data))
        print(f"save to {path}")


def loop_record(conn):
    while True:
        record_audio(conn)


def loop_asr(conn, download_root, model_size="base", target_lang="zh"):
    while True:
        msg, frames = conn.recv()
        if msg is None or msg == "stop":
            break
        print(f'Received: {msg}')
        # Convert the buffer frames to a NumPy array
        audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
        # Normalize the array to a [-1, 1] range
        audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        text = faster_whisper_transcribe(
            audio, download_root, model_size, target_lang)
        conn.send(("asr_text", text))
        g_text_dqueue.appendleft(text)


def loop_llm_generate(model_path):
    print("loop_llm_generate")
    while True:
        if g_text_dqueue:
            content = g_text_dqueue.pop()
            print(f"content {content}")
            generate(model_path, content)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', "-m", type=str,
                        default="./models", help='model root path')
    parser.add_argument('--model_size', "-s", type=str,
                        default="base", help='model size')
    parser.add_argument('--lang', "-l", type=str,
                        default="zh", help='target language')
    parser.add_argument('--llm_model_path', "-lm", type=str,
                        default="./models/Phi-3-mini-4k-instruct-q4.gguf", help='llm model path')
    args = parser.parse_args()

    parent_conn, child_conn = multiprocessing.Pipe()
    # p = multiprocessing.Process(target=record_audio, args=(child_conn,))
    p = multiprocessing.Process(target=loop_record, args=(child_conn,))
    c = multiprocessing.Process(target=loop_asr, args=(
        parent_conn, args.model_path, args.model_size, args.lang))
    p.start()
    c.start()
    asyncio.run(loop_llm_generate(args.llm_model_path))
    p.join()
    c.join()
