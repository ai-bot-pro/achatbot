import numpy as np


def generate_silence(duration=3, sr=44100):
    """
    生成空白音频文件
    """
    silence = np.zeros(int(duration * sr))

    return silence


def generate_white_noise(duration=3, sr=44100):
    """
    生成白噪声
    duration: 持续时间(秒)
    sr: 采样率
    """
    # 生成随机噪声
    samples = np.random.normal(0, 1, size=int(duration * sr))
    # 归一化到 [-1, 1] 范围
    samples = samples / np.max(np.abs(samples))

    return samples


def generate_random_sine(duration=3, sr=44100):
    """
    生成随机频率的正弦波
    """
    # 随机选择频率 (20Hz-2000Hz)
    freq = np.random.uniform(20, 2000)

    # 生成时间序列
    t = np.linspace(0, duration, int(sr * duration))

    # 生成正弦波
    samples = np.sin(2 * np.pi * freq * t)

    return samples


def generate_random_notes(duration=3, sr=44100):
    """
    生成随机音符序列
    """
    # 定义一些音符频率 (C4-C5)
    notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

    # 随机选择音符
    note_duration = 0.5  # 每个音符持续0.5秒
    num_notes = int(duration / note_duration)

    samples = np.array([])
    for _ in range(num_notes):
        freq = np.random.choice(notes)
        t = np.linspace(0, note_duration, int(sr * note_duration))
        note = np.sin(2 * np.pi * freq * t)
        # 添加淡入淡出效果
        note = note * np.hanning(len(note))
        samples = np.append(samples, note)

    # 归一化
    samples = samples / np.max(np.abs(samples))

    return samples


def generate_human_like_voice(duration=3, sr=44100):
    """
    生成类似人声的随机音频
    duration: 持续时间(秒)
    sr: 采样率
    """
    # 基础频率范围 (人声基频大约在85-255Hz之间)
    fundamental_freq = np.random.uniform(85, 255)

    # 生成时间序列
    t = np.linspace(0, duration, int(sr * duration))

    # 生成基础信号
    signal = np.zeros_like(t)

    # 添加基频和谐波
    for i in range(1, 6):
        # 每个谐波的振幅逐渐减小
        amplitude = 1.0 / i
        # 添加细微的频率变化模拟自然语音
        freq_variation = np.sin(2 * np.pi * 2 * t) * 10
        signal += amplitude * np.sin(2 * np.pi * (fundamental_freq * i + freq_variation) * t)

    # 添加音量包络
    num_syllables = int(duration * 2)  # 每秒约2个音节
    envelope = np.zeros_like(t)

    for i in range(num_syllables):
        center = i * (len(t) / num_syllables)
        width = len(t) / (num_syllables * 4)
        envelope += np.exp(-((t - center / sr) ** 2) / (2 * (width / sr) ** 2))

    envelope = envelope / np.max(envelope)
    signal = signal * envelope

    # 添加一些噪声模拟气流
    noise = np.random.normal(0, 0.01, len(signal))
    signal = signal + noise

    # 归一化
    signal = signal / np.max(np.abs(signal))

    # 应用低通滤波
    # signal = librosa.effects.preemphasis(signal, coef=0.95)

    return signal


"""
AUDIO_FUNCTION=generate_silence AUDIO_DURATION=3 AUDIO_SR=16000 python -m src.modules.speech.help.audio_mock 
AUDIO_FUNCTION=generate_white_noise AUDIO_DURATION=3 AUDIO_SR=16000 python -m src.modules.speech.help.audio_mock 
AUDIO_FUNCTION=generate_random_sine AUDIO_DURATION=3 AUDIO_SR=16000 python -m src.modules.speech.help.audio_mock 
AUDIO_FUNCTION=generate_random_notes AUDIO_DURATION=3 AUDIO_SR=16000 python -m src.modules.speech.help.audio_mock 
AUDIO_FUNCTION=generate_human_like_voice AUDIO_DURATION=3 AUDIO_SR=16000 python -m src.modules.speech.help.audio_mock 
"""

if __name__ == "__main__":
    import soundfile as sf
    import os
    from src.common.types import ASSETS_DIR

    duration = int(os.getenv("AUDIO_DURATION", "3"))  # 默认3秒
    sr = int(os.getenv("AUDIO_SR", "44100"))  # 默认采样率44100Hz
    function_name = os.getenv("AUDIO_FUNCTION", "generate_silence")
    save_path = os.path.join(ASSETS_DIR, f"{function_name}.wav")

    # 获取对应的函数对象
    function = globals().get(function_name)
    if function is None:
        raise ValueError(f"未找到函数: {function_name}")

    samples = function(duration, sr)
    sf.write(save_path, samples, sr)
    info = sf.info(save_path)
    print(info)
