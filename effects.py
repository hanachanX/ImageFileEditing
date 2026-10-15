"""モザイク・ドットアート・絵画風・ハーフトーンなどの効果。"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import block_means, expand_blocks, gray_to_rgb
from .geometry import normalize_box


def mosaic(img: np.ndarray, block: int, region=None) -> np.ndarray:
    """モザイク。block 画素四方ごとに平均色でぬりつぶす。

    region=(x0, y0, x1, y1) を渡すと、その矩形の中だけにかける（ブロックは矩形の左上から数える）。
    """
    block = max(1, int(block))
    out = img.copy()
    if region is None:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
    else:
        nb = normalize_box(region, img.shape[1], img.shape[0])
        if nb is None:
            raise ValueError("モザイクをかける範囲が空です")
        x0, y0, x1, y1 = nb
    sub = img[y0:y1, x0:x1]
    means = block_means(sub, block)
    filled = expand_blocks(np.rint(means), block, sub.shape[0], sub.shape[1])
    out[y0:y1, x0:x1] = np.clip(filled, 0, 255).astype(np.uint8)
    return out


def dot_art(img: np.ndarray, size: int = 10) -> np.ndarray:
    """ドットアート。size 画素四方ごとの平均色の丸を、黒い背景に並べる。"""
    size = max(3, int(size))
    h, w = img.shape[:2]
    means = block_means(img, size).astype(np.uint8)         # 小数部は切り捨て
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)
    for j in range(means.shape[0]):
        y = j * size
        for i in range(means.shape[1]):
            x = i * size
            draw.ellipse((x + 1, y + 1, x + size - 1, y + size - 1), fill=tuple(int(c) for c in means[j, i]))
    return np.asarray(canvas)


def painterly(img: np.ndarray, density: float = 1.0, max_radius: int = 8, seed=None) -> np.ndarray:
    """絵画風。ランダムな位置に、その位置の色の丸い点を、画素数×density 回だけ打つ。"""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = max(1, int(w * h * density))
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    rs = rng.integers(1, int(max_radius) + 1, n)
    colors = img[ys, xs]
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)
    for x, y, r, c in zip(xs.tolist(), ys.tolist(), rs.tolist(), colors.tolist()):
        draw.ellipse((x, y, x + r, y + r), fill=(c[0], c[1], c[2]))
    return np.asarray(canvas)


# ---------------------------------------------------------------- ハーフトーン（新聞写真風）
# 参考: https://github.com/SankarSrin/Digital-Halftoning
_BAYER_DISPERSED = np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21]], dtype=np.float32) / 64.0

_BAYER_CLUSTERED = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37],
    [8, 0, 2, 14, 45, 59, 61, 51],
    [22, 6, 4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [34, 46, 48, 36, 25, 11, 13, 27],
    [44, 58, 60, 50, 9, 1, 3, 15],
    [42, 56, 62, 52, 23, 7, 5, 17],
    [32, 40, 54, 38, 31, 21, 19, 29]], dtype=np.float32) / 64.0

_ULICHNEY_3 = np.array([
    [12, 5, 6, 13],
    [4, 0, 1, 7],
    [11, 3, 2, 8],
    [15, 10, 9, 14]]) / 15

_ULICHNEY_4 = np.array([
    [0.567, 0.635, 0.608, 0.514, 0.424, 0.365, 0.392, 0.486],
    [0.847, 0.878, 0.910, 0.698, 0.153, 0.122, 0.090, 0.302],
    [0.820, 0.969, 0.941, 0.667, 0.180, 0.031, 0.059, 0.333],
    [0.725, 0.788, 0.757, 0.545, 0.275, 0.212, 0.243, 0.455],
    [0.424, 0.365, 0.392, 0.486, 0.567, 0.635, 0.608, 0.514],
    [0.153, 0.122, 0.090, 0.302, 0.847, 0.878, 0.910, 0.698],
    [0.180, 0.031, 0.059, 0.333, 0.820, 0.969, 0.941, 0.667],
    [0.275, 0.212, 0.243, 0.455, 0.725, 0.788, 0.757, 0.545]])

_ULICHNEY_5 = np.array([
    [131, 187, 8, 78, 50, 18, 134, 89, 155, 102, 29, 95, 184, 73],
    [22, 86, 113, 171, 142, 105, 34, 166, 9, 60, 151, 128, 40, 110],
    [168, 137, 45, 28, 64, 188, 82, 54, 124, 189, 80, 13, 156, 56],
    [7, 61, 186, 121, 154, 6, 108, 177, 24, 100, 38, 176, 93, 123],
    [83, 148, 96, 17, 88, 133, 44, 145, 69, 161, 139, 72, 30, 181],
    [115, 27, 163, 47, 178, 65, 164, 14, 120, 48, 5, 127, 153, 52],
    [190, 58, 126, 81, 116, 21, 106, 77, 173, 92, 191, 63, 99, 12],
    [76, 144, 4, 185, 37, 149, 192, 39, 135, 23, 117, 31, 170, 132],
    [35, 172, 103, 66, 129, 79, 3, 97, 57, 159, 70, 141, 53, 94],
    [114, 20, 49, 158, 19, 146, 169, 122, 183, 11, 104, 180, 2, 165],
    [152, 87, 182, 118, 91, 42, 67, 25, 84, 147, 43, 85, 125, 68],
    [16, 136, 71, 10, 193, 112, 160, 138, 51, 111, 162, 26, 194, 46],
    [174, 107, 41, 143, 33, 74, 1, 101, 195, 15, 75, 140, 109, 90],
    [32, 62, 157, 98, 167, 119, 179, 59, 36, 130, 175, 55, 0, 150]]) / 196

HALFTONE_METHODS = {
    1: "Bayer 1（分散ドット）",
    2: "Bayer 2（集中ドット）",
    3: "Ulichney 1（4x4）",
    4: "Ulichney 2（8x8）",
    5: "Ulichney 3（14x14）",
}


def _tile_to(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    """しきい値の模様を、画像の大きさまで敷きつめる（余りは切り落とす）。"""
    ph, pw = pattern.shape
    reps = (-(-height // ph), -(-width // pw))
    return np.tile(pattern, reps)[:height, :width]


def halftone(img: np.ndarray, method: int) -> np.ndarray:
    """ハーフトーン（白と黒の2値）。method は HALFTONE_METHODS の番号。"""
    h, w = img.shape[:2]
    if method in (1, 2):
        gray = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
        mask = _tile_to(_BAYER_DISPERSED if method == 1 else _BAYER_CLUSTERED, h, w)
        white = gray >= mask
    elif method in (3, 4, 5):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        pattern = {3: _ULICHNEY_3, 4: _ULICHNEY_4, 5: _ULICHNEY_5}[method]
        mask = _tile_to(pattern.T, h, w)
        white = gray > mask
    else:
        raise ValueError(f"ハーフトーンの方式が正しくありません: {method}")
    return gray_to_rgb(white.astype(np.uint8) * 255)
