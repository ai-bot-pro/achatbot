# -*- coding: utf-8 -*-
"""
实时语音识别（Whisper/Faster-Whisper）示例

功能:
- 使用 sounddevice 从麦克风实时采集 16kHz PCM 数据
- 以 5s 为单位采集音频块，交给 Whisper 做转录
- 支持 CPU 或 CUDA（若可用），默认加载 small 模型（可以改成 tiny/base/large）
- 结果实时打印（可自行改为 GUI、WebSocket 等）

运行前:
  pip install torch "openai-whisper" faster-whisper sounddevice numpy scipy

粗粒度的近实时ASR
"""

import os
import queue
import threading
import time
from typing import List

import numpy as np
import sounddevice as sd
import torch

# ------------------------------- 参数配置 -------------------------------

# 采样率 & 格式 (Whisper 预期 16kHz, 16-bit mono)
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 5.0  # seconds，每块音频长度, 5s 足够生成一段可识别的句子
OVERLAP = 0.5  # seconds，拼接时保留的重叠区域，防止跨块单词截断

# Whisper模型
# 1. official:   import whisper ; model = whisper.load_model("small")
# 2. faster-whisper (推荐): from faster_whisper import WhisperModel
USE_FAST = True  # 设为 True 使用 faster-whisper (速度 3‑5 倍)
MODEL_SIZE = "small"  # tiny / base / small / medium / large

# 是否使用 GPU（如果 CUDA 可用）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------- 模型加载 ----------------------------
print(f"Loading Whisper model ({MODEL_SIZE}) on {DEVICE} ...")
if USE_FAST:
    from faster_whisper import WhisperModel  # pip install faster-whisper

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")
else:
    import whisper

    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# --------------------------------- 队列 --------------------------------
# 用于在采集线程与识别线程之间传递音频块 (numpy.ndarray)
audio_queue = queue.Queue(maxsize=5)  # 限制队列长度防止内存膨胀


# --------------------------------- 采集函数 -------------------------
def audio_callback(indata, frames, time_info, status):
    """
    sounddevice 回调函数，每次捕获的是一块 (frames,) 的 numpy 语音片段.
    把采样转换为 float32（-1~1），并放入共享队列。
    """
    if status:
        print("[Audio] Warning:", status)

    # 把 int16 -> float32
    audio_chunk = indata[:, 0].astype(np.float32)  # 只取单声道
    # 放入队列，如果满了则丢掉最旧的
    try:
        audio_queue.put_nowait(audio_chunk)
    except queue.Full:
        # 队列满，丢旧帧（实时要求不保留旧数据）
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        audio_queue.put_nowait(audio_chunk)


def start_capture():
    """打开音频输入流，开启采集线程。"""
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),  # 每次收满一个 block
        callback=audio_callback,
    )
    stream.start()
    print(f"[Info] 已打开音频输入（采样率 {SAMPLE_RATE} Hz）")
    return stream


# -------------------------- 停止/退出处理 -------------------------
def soft_exit():
    """在 ctrl-c 时安全退出。"""
    global running
    running = False
    print("\n[Info] 正在退出...")


# --------------------------- 识别主循环 ------------------------
def recognize_loop():
    """
    轮询音频队列，用滑动窗口把连续 chunk 拼成完整块
    （含 0.5s 重叠），发送给 Whisper，打印结果。
    """
    # 缓冲区 (list of np.ndarray)
    buffer: List[np.ndarray] = []
    last_timestamp = None  # 用于对齐窗口的时间戳（可选）

    while running:
        try:
            # 取出一段 audio, 规避阻塞太久
            chunk = audio_queue.get(timeout=0.1)  # 获取最新块
            buffer.append(chunk)
            # 合并至一定长度后（>= BLOCK_DURATION），开始处理
            # 拼接后再做一次切分，保留前后 OVERLAP 秒重叠
            # 一次处理的时长 = BLOCK_DURATION + OVERLAP
            # 这里使用简单的累加，实际可使用更精细的 VAD 剪切
        except queue.Empty:
            # queue 空，啥都不干
            continue

        # 检查累计时长
        total_len = sum(arr.shape[0] for arr in buffer) / SAMPLE_RATE
        if total_len < BLOCK_DURATION:
            continue

        # 合并为一个完整的 numpy array (float32)
        audio = np.concatenate(buffer, axis=0)  # shape (samples,)
        # 清空 buffer，为下一轮做准备（保留 OVERLAP 区域）
        # 先复制出要保留的尾部
        overlap_samples = int(SAMPLE_RATE * OVERLAP)
        if overlap_samples > 0:
            tail = audio[-overlap_samples:].copy()
        else:
            tail = np.array([], dtype=np.float32)

        # 清理 buffer，保留 tail
        buffer = [tail] if tail.shape[0] > 0 else []
        # 我们对完整音段做一次转写
        # Whisper 接收的是 array 或 (speech,) 的 numpy，或 torch 媒体
        # >>> 为确保兼容，强制将 audio 归一化为 -1 ~ 1
        audio = np.clip(audio / np.max(np.abs(audio) + 1e-10), -1.0, 1.0)

        # Whisper 需要 16-bit PCM (int16) 或 float32，两个都行
        # 使用 faster-whisper 的接口
        if USE_FAST:
            # faster-whisper 支持直接传 numpy (float32)
            segments, info = model.transcribe(
                audio,
                beam_size=5,
                language="zh",  # 中文；设为 None 自动检测
                word_timestamps=False,  # 若要 word level，可打开
                # vad = False (默认)，若要 VAD 可自行启用
                # 在这里可以传更多参数，参考 doc
                # context: optional 对话上下文
                vad_filter=False,
                force_full_segments=True,
                patience=1.0,
            )
            # segments 是迭代器，内部已包含时间戳
            # 只关注文本
            text = " ".join([s.text for s in segments])
        else:
            # 官方 whisper 的接口
            # 需要 file 或 audio array
            result = model.transcribe(
                audio,
                language="zh",  # 把语言固定为中文；若不确定，传 None
                fp16=torch.cuda.is_available(),
                # 如果 want timestamps:
                # without timestamps: **segments** includes start and end.
            )
            text = result["text"].strip()

        # 打印结果
        if text:
            # 用 timestamp (相对本次 block 的开始)
            now = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{now}] 识别结果: {text}")


# ----------------------- 主程序入口 ---------------------------
if __name__ == "__main__":
    # 捕获 Ctrl-C 退出
    import signal

    signal.signal(signal.SIGINT, lambda s, f: soft_exit())

    # 启动音频采集
    stream = None
    try:
        stream = start_capture()
        # 主循环在子线程里跑
        running = True
        # 开启识别线程（保持 main 线程干净）
        rt_thread = threading.Thread(target=recognize_loop, daemon=True)
        rt_thread.start()

        # 主线程只负责保持进程活着，阻塞等待 Ctrl-C
        while running:
            time.sleep(0.5)
    finally:
        # 关闭采集流（确保释放声卡资源）
        if stream:
            stream.stop()
            stream.close()
        print("[Info] 程序结束，已释放音频资源")
