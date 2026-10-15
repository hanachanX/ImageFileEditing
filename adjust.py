"""色・明るさの調整。"""
from __future__ import annotations

import cv2
import numpy as np

from .common import gray_to_rgb

_SEPIA = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131],
])


def gamma(img: np.ndarray, value: float) -> np.ndarray:
    """ガンマ補正。value が大きいほど明るくなる（value=1.0 で変化なし）。"""
    if value <= 0:
        raise ValueError("ガンマ値は0より大きくしてください")
    out = np.power(img / 255.0, 1.0 / value) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """コントラスト。alpha=1.0 で変化なし。"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0.0)


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """明るさ。delta（-255〜255）を全画素に足す。"""
    out = np.clip(img.astype(np.float32) + delta, 0, 255)
    return out.astype(np.uint8)


def rgb_gain(img: np.ndarray, r: float, g: float, b: float) -> np.ndarray:
    """赤・緑・青のチャンネルごとに倍率をかける（1.0 で変化なし）。"""
    a = img.astype(np.float32) / 255.0
    a[:, :, 0] *= r
    a[:, :, 1] *= g
    a[:, :, 2] *= b
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def hsv_adjust(img: np.ndarray, hue_shift: float = 0.0, sat_scale: float = 1.0, val_shift: float = 0.0) -> np.ndarray:
    """色相・彩度・明度の調整。

    hue_shift: 色相を回す角度（度。0〜360 で一周）
    sat_scale: 彩度にかける倍率（1.0 で変化なし）
    val_shift: 明度に足す値（-255〜255）
    """
    # 8bit のまま HSV にすると色相が 0..179 に丸められて劣化するので、小数（float32）で計算する
    hsv = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)   # H: 0..360, S,V: 0..1
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 360.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift / 255.0, 0.0, 1.0)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def grayscale(img: np.ndarray) -> np.ndarray:
    return gray_to_rgb(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))


def sepia(img: np.ndarray) -> np.ndarray:
    out = np.dot(img, _SEPIA.T)
    return np.clip(out, 0, 255).astype(np.uint8)


def posterize(img: np.ndarray, levels: int) -> np.ndarray:
    """減色。各チャンネルを levels 段階にする。"""
    levels = max(2, int(levels))
    size = 256 / levels
    q = (img / size).astype(np.uint8)
    return (q * size + size / 2).astype(np.uint8)
