"""ぼかし・シャープ・輪郭などのフィルタ。"""
from __future__ import annotations

import cv2
import numpy as np

from .common import gray_to_rgb


def gaussian_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """ガウシアンぼかし。kernel は奇数にそろえる（偶数なら +1）。sigma=0 なら kernel から自動で決まる。"""
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def sharpen(img: np.ndarray, amount: float = 0.5, radius: float = 3.0) -> np.ndarray:
    """アンシャープマスク。amount=0 で変化なし。"""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def dilate(img: np.ndarray, iterations: int = 1) -> np.ndarray:
    """膨張（明るい部分が広がる）。"""
    return cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=int(iterations))


def erode(img: np.ndarray, iterations: int = 1) -> np.ndarray:
    """収縮（暗い部分が広がる）。"""
    return cv2.erode(img, np.ones((3, 3), np.uint8), iterations=int(iterations))


def clahe(img: np.ndarray, clip_limit: float = 3.0, tiles: int = 8) -> np.ndarray:
    """CLAHE（局所的なヒストグラム平坦化）。明るさ成分だけに適用する。"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(tiles), int(tiles))).apply(l_ch)
    return cv2.cvtColor(cv2.merge((eq, a_ch, b_ch)), cv2.COLOR_Lab2RGB)


def pencil_sketch(img: np.ndarray) -> np.ndarray:
    """鉛筆画風。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    smooth = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    dodge = cv2.divide(gray, 255 - smooth, scale=256)
    return gray_to_rgb(dodge)
