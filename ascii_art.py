"""アスキーアート（全角文字）への変換。表示は画面側が行う。"""
from __future__ import annotations

import numpy as np

from .common import block_sums

# 暗い→明るい の順に並べた文字。
CHARSET = 'ＭＷＮ＄＠％＃＆Ｂ８９ＥＧＡ６ｍＫ５ＨＲｋｂＹＴ４３Ｖ０ＪＬ７ｇｐａｓｅｙｘｚｎｏｃｖ？ｊＩｆｔｒ１ｌｉ＊＝－～＾｀’：；，．　'[::-1]

MIN_COLUMNS, MAX_COLUMNS = 40, 200


def to_ascii(img: np.ndarray, columns: int) -> str:
    """画像を、おおよそ columns 文字の幅のアスキーアートにする。"""
    if not (MIN_COLUMNS <= columns <= MAX_COLUMNS):
        raise ValueError(f"幅は{MIN_COLUMNS}〜{MAX_COLUMNS}の整数にしてください")
    h, w = img.shape[:2]
    block = max(1, w // columns)
    sums, count = block_sums(img, block)
    mean = sums.sum(axis=2) / (count * img.shape[2])          # 3チャンネルをまとめた平均
    n = len(CHARSET)
    index = np.clip((mean / 255 * (n - 1)).astype(int), 0, n - 1)
    return "\n".join("".join(CHARSET[i] for i in row) for row in index) + "\n"
