import base64
from functools import lru_cache
import numpy as np


def image_bytes_to_base64_data_uri(img_bytes: bytes, format: str = "jpeg", encoding="utf-8"):
    base64_data = base64.b64encode(img_bytes).decode(encoding)
    return f"data:image/{format};base64,{base64_data}"


def image_to_base64_data_uri(file_path: str, format: str = "jpeg", encoding="utf-8"):
    with open(file_path, "rb") as img_file:
        return image_bytes_to_base64_data_uri(img_file.read(), format=format, encoding=encoding)


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert YUV to RGB using numpy operations"""
    # YUV to RGB conversion matrix
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    # RGB conversion using BT.601 standard
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u

    # Clip values to [0, 255] and convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def resize_plane(plane: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize UV plane using bilinear interpolation"""
    h, w = plane.shape
    target_h, target_w = target_shape

    # Create coordinate matrices
    x = np.linspace(0, w - 1, target_w)
    y = np.linspace(0, h - 1, target_h)
    x_coords, y_coords = np.meshgrid(x, y)

    # Get integer and fractional parts
    x0 = np.floor(x_coords).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y_coords).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)

    # Get weights
    wx = x_coords - x0
    wy = y_coords - y0

    # Get values at corners
    v00 = plane[y0, x0]
    v10 = plane[y1, x0]
    v01 = plane[y0, x1]
    v11 = plane[y1, x1]

    # Interpolate
    result = v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy

    return result.astype(np.uint8)


def nv21_to_rgb(y_data: np.ndarray, vu_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert NV21 to RGB using numpy operations
    NV21格式: YYYYYYYY VUVU (V和U交错排列)
    Y: 亮度平面
    VU: V和U交错的色度平面
    """
    # 重塑Y平面和VU平面
    y = y_data.reshape(height, width)
    vu = vu_data.reshape(height // 2, width // 2, 2)

    # 分离V和U通道并放大到全尺寸
    v = np.repeat(np.repeat(vu[:, :, 0], 2, axis=0), 2, axis=1)
    u = np.repeat(np.repeat(vu[:, :, 1], 2, axis=0), 2, axis=1)

    # 转换为float进行计算
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    # YUV到RGB的转换 (使用BT.601标准)
    r = y + 1.370 * v
    g = y - 0.698 * v - 0.336 * u
    b = y + 1.732 * u

    # 合并通道并裁剪到[0, 255]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


@lru_cache(maxsize=8)
def _get_conversion_matrices(height: int, width: int) -> tuple:
    """缓存转换矩阵以提高性能"""
    # 创建用于重复的索引矩阵
    y_indices = np.arange(height * width).reshape(height, width)
    vu_indices = np.arange(height * width // 4).reshape(height // 2, width // 2)
    return y_indices, vu_indices


def nv21_to_rgb_optimized(
    y_data: np.ndarray, vu_data: np.ndarray, width: int, height: int
) -> np.ndarray:
    """优化的NV21到RGB转换"""
    # 使用缓存的转换矩阵
    y_indices, vu_indices = _get_conversion_matrices(height, width)

    # 使用内存视图和预分配的数组
    y = np.asarray(memoryview(y_data)).reshape(height, width)
    vu = np.asarray(memoryview(vu_data)).reshape(height // 2, width // 2, 2)

    # 使用向量化操作进行色度平面的放大
    v = vu[y_indices // 2 // width, y_indices // 2 % width, 0]
    u = vu[y_indices // 2 // width, y_indices // 2 % width, 1]

    # 转换为float32进行计算
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    # 使用向量化操作进行颜色转换
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(y + 1.370 * v, 0, 255)  # R
    rgb[..., 1] = np.clip(y - 0.698 * v - 0.336 * u, 0, 255)  # G
    rgb[..., 2] = np.clip(y + 1.732 * u, 0, 255)  # B

    return rgb


def nv12_to_rgb(y_data: np.ndarray, uv_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert NV12 to RGB using numpy operations
    NV12格式: YYYYYYYY UVUV (U和V交错排列)
    Y: 亮度平面
    UV: U和V交错的色度平面
    """
    # 重塑Y平面和UV平面
    y = y_data.reshape(height, width)
    uv = uv_data.reshape(height // 2, width // 2, 2)

    # 分离U和V通道并放大到全尺寸
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)  # U在前
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)  # V在后

    # 转换为float进行计算
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    # YUV到RGB的转换 (使用BT.601标准)
    r = y + 1.370 * v
    g = y - 0.698 * v - 0.336 * u
    b = y + 1.732 * u

    # 合并通道并裁剪到[0, 255]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


def nv12_to_rgb_optimized(
    y_data: np.ndarray, uv_data: np.ndarray, width: int, height: int
) -> np.ndarray:
    """优化的NV12到RGB转换"""
    # 使用缓存的转换矩阵
    y_indices, uv_indices = _get_conversion_matrices(height, width)

    # 使用内存视图和预分配的数组
    y = np.asarray(memoryview(y_data)).reshape(height, width)
    uv = np.asarray(memoryview(uv_data)).reshape(height // 2, width // 2, 2)

    # 使用向量化操作进行色度平面的放大
    u = uv[y_indices // 2 // width, y_indices // 2 % width, 0]  # U在前
    v = uv[y_indices // 2 // width, y_indices // 2 % width, 1]  # V在后

    # 转换为float32进行计算
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    # 使用向量化操作进行颜色转换并预分配输出数组
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(y + 1.370 * v, 0, 255)  # R
    rgb[..., 1] = np.clip(y - 0.698 * v - 0.336 * u, 0, 255)  # G
    rgb[..., 2] = np.clip(y + 1.732 * u, 0, 255)  # B

    return rgb
