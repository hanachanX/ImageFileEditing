"""切り抜き・拡大縮小・反転・回転。"""
from __future__ import annotations

import cv2
import numpy as np

MAX_PIXELS = 120_000_000        # これを超える拡大は、メモリ不足を避けるため断る


def normalize_box(box, width: int, height: int):
    """ドラッグした矩形 (x0, y0, x1, y1) を、左上→右下の順に直して画像内に収める。

    面積が0になる場合は None を返す。
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


def crop(img: np.ndarray, box) -> np.ndarray:
    nb = normalize_box(box, img.shape[1], img.shape[0])
    if nb is None:
        raise ValueError("切り抜く範囲が空です")
    x0, y0, x1, y1 = nb
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def resize(img: np.ndarray, factor: float) -> np.ndarray:
    """倍率で拡大縮小（Lanczos）。"""
    if factor <= 0:
        raise ValueError("倍率は0より大きくしてください")
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    if new_w * new_h > MAX_PIXELS:
        raise ValueError(f"大きすぎます（{new_w}x{new_h}）。倍率を下げてください")
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1])


def flip_vertical(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, :])


def rotate_right(img: np.ndarray) -> np.ndarray:
    """時計回りに90度。"""
    return np.ascontiguousarray(np.rot90(img, k=-1))


def rotate_left(img: np.ndarray) -> np.ndarray:
    """反時計回りに90度。"""
    return np.ascontiguousarray(np.rot90(img, k=1))
