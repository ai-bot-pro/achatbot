import collections
import webrtcvad
import pyaudio
import wave


# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1600
FRAME_DURATION = 30  # 毫秒
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)

audio = pyaudio.PyAudio()
# for audio recording
stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)


def export_wav(data, filename):
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(data))
    wf.close()


def record_audio():
    frames = collections.deque(maxlen=30)  # 保存最近 30 个帧
    tmp = collections.deque(maxlen=1000)
    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 敏感度，0 到 3，0 最不敏感，3 最敏感
    triggered = False
    frames.clear()
    active_ratio = 0.7
    silent_ratio = 0.5
    frame_size = FRAME_SIZE
    print(f"read audio frame size: {frame_size}")
    while True:
        frame = stream.read(frame_size)
        is_speech = vad.is_speech(frame, RATE)
        if is_speech:
            tmp.append(frame)
        if not triggered:
            frames.append((frame, is_speech))
            num_voiced = len([f for f, speech in frames if speech])
            print(f"num_voiced {num_voiced}")
            if num_voiced > active_ratio * frames.maxlen:
                print("start recording...")
                triggered = True
                frames.clear()
        else:
            frames.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in frames if not speech])
            print(f"num_unvoiced {num_voiced}")
            if num_unvoiced > silent_ratio * frames.maxlen:
                print("stop recording...")
                export_wav(tmp, "./records/tmp_webrtcvad.wav")
                break


if __name__ == "__main__":
    record_audio()
