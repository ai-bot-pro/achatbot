import json
import os


async def save_audio_to_file(audio_data, file_name, audio_dir="records", channles=1, sample_width=2, sample_rate=16000):
    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    import wave
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channles)  # Assuming mono audio
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return file_path


async def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)


async def get_audio_segment(file_path, start=None, end=None):
    from pydub import AudioSegment
    with open(file_path, 'rb') as file:
        audio = AudioSegment.from_file(file, format="wav")
    if start is not None and end is not None:
        # pydub works in milliseconds
        return audio[start * 1000:end * 1000]
    return audio
