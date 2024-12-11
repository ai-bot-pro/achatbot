import logging

import numpy as np
from PIL import Image
from agora.rtc.video_frame_observer import VideoFrame
from agora.rtc.video_frame_sender import ExternalVideoFrame

from src.common.utils.img_utils import *

"""
I420和I422的主要区别:
I420(YUV420P): U和V平面的宽度和高度都是Y平面的一半
I422(YUV422P): U和V平面的宽度是Y平面的一半,但高度相同

NV21格式特点:
Y平面: 存储亮度信息
VU平面: 交错存储色度信息(先V后U)
NV12与NV21的区别:
NV12: YYYYYYYY UVUV (U在前,V在后)
NV21: YYYYYYYY VUVU (V在前,U在后)
"""

# ------- VideoFrame to PIL.Image Image -------


def convert_I420_to_RGB(video_frame: VideoFrame) -> Image.Image:
    # YUV420P(I420) to RGB
    # TODO:
    # 使用numba进行JIT编译
    # 实现SIMD优化
    # 使用多线程处理不同的颜色平面 (or use cuda)

    width = video_frame.width
    height = video_frame.height

    # Extract YUV planes
    y_plane = np.frombuffer(video_frame.y_buffer, dtype=np.uint8).reshape(height, width)
    u_plane = np.frombuffer(video_frame.u_buffer, dtype=np.uint8).reshape(height // 2, width // 2)
    v_plane = np.frombuffer(video_frame.v_buffer, dtype=np.uint8).reshape(height // 2, width // 2)

    # Resize U and V planes
    u_resized = resize_plane(u_plane, (height, width))
    v_resized = resize_plane(v_plane, (height, width))

    # Convert to RGB
    rgb = yuv_to_rgb(y_plane, u_resized, v_resized)
    image = Image.fromarray(rgb)

    return image


def convert_I422_to_RGB(video_frame: VideoFrame) -> Image.Image:
    # YUV422(I422) to RGB
    width = video_frame.width
    height = video_frame.height

    # Extract YUV planes
    y_plane = np.frombuffer(video_frame.y_buffer, dtype=np.uint8).reshape(height, width)
    u_plane = np.frombuffer(video_frame.u_buffer, dtype=np.uint8).reshape(height, width // 2)
    v_plane = np.frombuffer(video_frame.v_buffer, dtype=np.uint8).reshape(height, width // 2)

    # Resize U and V planes (only width needs to be doubled)
    u_resized = resize_plane(u_plane, (height, width))
    v_resized = resize_plane(v_plane, (height, width))

    # Convert to RGB
    rgb = yuv_to_rgb(y_plane, u_resized, v_resized)
    image = Image.fromarray(rgb)

    return image


def convert_I420_to_RGB_with_cv(video_frame: VideoFrame) -> Image.Image:
    import cv2

    # 从YUV420P(I420)转换到RGB
    width = video_frame.width
    height = video_frame.height

    # 从YUV缓冲区提取Y、U、V平面
    # y_size = width * height
    # u_size = (width * height) // 4

    y_plane = np.frombuffer(video_frame.y_buffer, dtype=np.uint8).reshape(height, width)
    u_plane = np.frombuffer(video_frame.u_buffer, dtype=np.uint8).reshape(height // 2, width // 2)
    v_plane = np.frombuffer(video_frame.v_buffer, dtype=np.uint8).reshape(height // 2, width // 2)

    # 将U和V平面放大到与Y相同的尺寸
    u_resized = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

    # 将YUV转换为RGB
    yuv = cv2.merge([y_plane, u_resized, v_resized])
    # 如果出现图像质量问题,可以尝试调整插值方法(如改用cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    # 转换为PIL Image
    image = Image.fromarray(rgb)
    return image


def convert_I422_to_RGB_with_cv(video_frame: VideoFrame) -> Image.Image:
    import cv2

    # YUV422(I422) to RGB
    width = video_frame.width
    height = video_frame.height

    # y_size = width * height
    # U和V平面在I422中是Y平面的一半宽度,但高度相同
    # uv_size = (width * height) // 2

    # 从缓冲区提取Y、U、V平面
    y_plane = np.frombuffer(video_frame.y_buffer, dtype=np.uint8).reshape(height, width)
    u_plane = np.frombuffer(video_frame.u_buffer, dtype=np.uint8).reshape(height, width // 2)
    v_plane = np.frombuffer(video_frame.v_buffer, dtype=np.uint8).reshape(height, width // 2)

    # 将U和V平面放大到与Y相同的宽度
    u_resized = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

    # 合并YUV通道并转换为RGB
    yuv = cv2.merge([y_plane, u_resized, v_resized])
    # 如果出现图像质量问题,可以尝试调整插值方法(如改用cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    image = Image.fromarray(rgb)

    return image


def convert_RGBA_to_RGB(video_frame: VideoFrame) -> Image.Image:
    width = video_frame.width
    height = video_frame.height

    if video_frame.alpha_buffer is None:
        raise Exception("RGBA frame has no alpha_buffer")

    # 使用内存视图避免数据复制
    rgba_view = memoryview(video_frame.alpha_buffer)
    rgba_array = np.asarray(rgba_view, dtype=np.uint8)

    # 直接重塑并切片,避免额外的内存分配
    rgb_array = rgba_array.reshape(height, width, 4)[:, :, :3]

    image = Image.fromarray(rgb_array, mode="RGB")

    return image


def convert_NV21_to_RGB(video_frame: VideoFrame) -> Image.Image:
    width = video_frame.width
    height = video_frame.height
    # 从Y缓冲区获取亮度数据
    y_array = np.frombuffer(video_frame.y_buffer, dtype=np.uint8)

    # 从UV缓冲区获取色度数据
    # NV21中u_buffer存储VU数据
    vu_array = np.frombuffer(video_frame.u_buffer, dtype=np.uint8)

    # 转换为RGB
    rgb_array = nv21_to_rgb(y_array, vu_array, width, height)

    image = Image.fromarray(rgb_array, mode="RGB")

    return image


def convert_NV21_to_RGB_optimized(video_frame: VideoFrame) -> Image.Image:
    width = video_frame.width
    height = video_frame.height

    rgb_array = nv21_to_rgb_optimized(
        video_frame.y_buffer,
        video_frame.u_buffer,  # NV21中u_buffer存储VU数据
        width,
        height,
    )
    image = Image.fromarray(rgb_array, mode="RGB")

    return image


def convert_NV12_to_RGB(video_frame: VideoFrame) -> Image.Image:
    width = video_frame.width
    height = video_frame.height
    # 从Y缓冲区获取亮度数据
    y_array = np.frombuffer(video_frame.y_buffer, dtype=np.uint8)

    # 从UV缓冲区获取色度数据
    # NV12中u_buffer存储VU数据
    uv_array = np.frombuffer(video_frame.u_buffer, dtype=np.uint8)

    # 转换为RGB
    rgb_array = nv12_to_rgb(y_array, uv_array, width, height)

    image = Image.fromarray(rgb_array, mode="RGB")

    return image


def convert_NV12_to_RGB_optimized(video_frame: VideoFrame) -> Image.Image:
    width = video_frame.width
    height = video_frame.height

    rgb_array = nv12_to_rgb_optimized(
        video_frame.y_buffer,
        video_frame.u_buffer,  # NV12中u_buffer存储VU数据
        width,
        height,
    )
    image = Image.fromarray(rgb_array, mode="RGB")

    return image


# ------- PIL.Image Image to ExternalVideoFrame -------
