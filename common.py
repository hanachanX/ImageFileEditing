"""画像処理で共通して使う小さな部品。"""
from __future__ import annotations

import numpy as np


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """どんな配列が来ても、(H, W, 3) の uint8 にそろえる。"""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"画像の形式が正しくありません: shape={a.shape}")
    return np.ascontiguousarray(a)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """1チャンネルの画像を、見た目はそのままで3チャンネルにする。"""
    g = np.asarray(gray)
    return np.ascontiguousarray(np.stack([g, g, g], axis=2))


def block_sums(a: np.ndarray, block: int):
    """画像を block×block の正方形に区切り、区切りごとの合計と画素数を返す。

    右端・下端で余った小さなブロックも、その大きさのまま正しく数える。
    戻り値: (合計 [ny, nx, C], 画素数 [ny, nx])
    """
    h, w = a.shape[:2]
    ys = np.arange(0, h, block)
    xs = np.arange(0, w, block)
    s = np.add.reduceat(a.astype(np.float64), ys, axis=0)
    s = np.add.reduceat(s, xs, axis=1)
    ch = np.diff(np.append(ys, h))
    cw = np.diff(np.append(xs, w))
    count = ch[:, None] * cw[None, :]
    return s, count


def block_means(a: np.ndarray, block: int) -> np.ndarray:
    """block×block ごとの平均値 [ny, nx, C]（float64）。"""
    s, count = block_sums(a, block)
    return s / count[:, :, None]


def expand_blocks(small: np.ndarray, block: int, height: int, width: int) -> np.ndarray:
    """ブロックごとの値を、元の大きさに引き伸ばす（右端・下端は切り落とす）。"""
    big = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    return big[:height, :width]
