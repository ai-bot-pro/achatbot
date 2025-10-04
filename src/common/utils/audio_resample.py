import torch
import torchaudio
import numpy as np
import librosa


def resample_file(
    input_path: str,
    output_path: str,
    new_freq: int = 48000,
):
    """
    默认 将16K采样率音频转换为48K采样率

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径 (默认48K采样率)
    """
    waveform, sample_rate = torchaudio.load(input_path)
    if sample_rate == new_freq:
        return

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_freq,
        resampling_method="kaiser_window",  # 高质量重采样算法
        # resampling_method="sinc_interpolation",
    )

    # 执行重采样
    resampled_waveform = resampler(waveform)

    torchaudio.save(
        output_path,
        resampled_waveform,
        new_freq,
        bits_per_sample=16,  # 保持16位深度
    )


def resample_bytes2bytes(
    audio_bytes: bytes, orig_freq: int = 16000, new_freq: int = 48000
) -> bytes:
    """
    默认 将16K采样率的音频字节流转换为48K采样率的音频字节流

    参数:
        audio_bytes: 默认16K采样率的原始PCM音频字节流 (16位深度, 单声道)

    返回:
        bytes: 默认48K采样率的PCM音频字节流
    """
    # 将字节流转换为PyTorch张量
    waveform = torch.frombuffer(audio_bytes, dtype=torch.int16).float()
    waveform = waveform.unsqueeze(0)  # 添加通道维度 (1, N)

    # 创建重采样器 (16K → 48K)
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_freq, new_freq=new_freq, resampling_method="sinc_interp_kaiser"
    )

    # 执行重采样
    resampled_waveform = resampler(waveform)

    # 转换回16位整数并移除通道维度
    resampled_waveform = resampled_waveform.squeeze(0).to(torch.int16)

    # 将张量转换回字节流
    return resampled_waveform.numpy().tobytes()


def resample_bytes2numpy(
    audio_bytes: bytes, orig_freq: int = 16000, new_freq: int = 48000
) -> np.ndarray:
    """
    默认 将16K采样率的音频字节流转换为48K采样率的音频字节流

    参数:
        audio_bytes: 默认16K采样率的原始PCM音频字节流 (16位深度, 单声道)

    返回:
        np.ndarray : 默认48K采样率的PCM音频数据（16位深度, 单声道）
    """
    # 1. 转换为 int16 数组
    waveform_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # 2. 转换为归一化浮点数
    waveform_float = waveform_int16.astype(np.float32) / 32768.0
    # 3. 确保二维形状 (1, N)
    waveform_float = np.atleast_2d(waveform_float)
    if orig_freq != new_freq:
        # 4. 重采样 orig→target
        # https://librosa.org/doc/latest/generated/librosa.resample.html
        waveform_float = librosa.resample(
            waveform_float,
            orig_sr=orig_freq,
            target_sr=new_freq,
            # https://python-soxr.readthedocs.io/en/latest/soxr.html#module-soxr
            # res_type="soxr_vhq",
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample
            # res_type="scipy", # (or "fft") Fourier method.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
            # res_type="polyphase", # fast but low quality
            # https://resampy.readthedocs.io/en/stable/api.html#module-resampy
            res_type="kaiser_best",  # slow but high quality
        )
    # 5. 转换回16位整数 for audio bytes
    waveform_target_int16 = (waveform_float * 32767).astype(np.int16)

    return waveform_target_int16


def resample_numpy2bytes(
    waveform_float: np.ndarray, orig_freq: int = 16000, new_freq: int = 48000
) -> bytes:
    """
    默认 将16K采样率的音频字节流转换为48K采样率的音频字节流

    参数:
        np.ndarray : 默认16K采样率的PCM音频数据（16位深度, 单声道）

    返回:
        audio_bytes: 默认48K采样率的原始PCM音频字节流 (16位深度, 单声道)
    """
    # 3. 确保二维形状 (1, N)
    waveform_float = np.atleast_2d(waveform_float)
    if orig_freq != new_freq:
        # 4. 重采样 orig→target
        # https://librosa.org/doc/latest/generated/librosa.resample.html
        waveform_float = librosa.resample(
            waveform_float,
            orig_sr=orig_freq,
            target_sr=new_freq,
            # https://python-soxr.readthedocs.io/en/latest/soxr.html#module-soxr
            # res_type="soxr_vhq",
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample
            # res_type="scipy", # (or "fft") Fourier method.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
            # res_type="polyphase", # fast but low quality
            # https://resampy.readthedocs.io/en/stable/api.html#module-resampy
            res_type="kaiser_best",  # slow but high quality
        )
    # 5. 转换回16位整数 for audio bytes
    waveform_target_int16 = (waveform_float * 32767).astype(np.int16)

    return waveform_target_int16.tobytes()
