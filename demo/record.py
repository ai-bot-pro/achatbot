import wave
import struct
import pyaudio

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
# two seconds of silence marks the end of user voice input
SILENT_CHUNKS = 2 * RATE / CHUNK
# Set microphone id. Use tools/list_microphones.py to see a device list.
MIC_IDX = 0


def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i**2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=None,
        frames_per_buffer=CHUNK,
    )

    silent_chunks = 0
    audio_started = False
    frames = []

    print("start recording")
    while True:
        data = stream.read(CHUNK)
        rms = compute_rms(data)
        print(f"rms:{rms} silence threshold:{SILENCE_THRESHOLD}")
        if audio_started:
            frames.append(data)
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True
            frames.append(data)

    print("stop recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open("records/tmp.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()


if __name__ == "__main__":
    record_audio()
