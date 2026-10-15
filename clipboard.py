"""クリップボードとの画像のやりとり。"""
from __future__ import annotations

import io
import sys

import numpy as np
from PIL import Image, ImageGrab


def get_image() -> np.ndarray | None:
    """クリップボードの画像を RGB 配列で返す。画像が無ければ None。"""
    data = ImageGrab.grabclipboard()
    if isinstance(data, Image.Image):
        return np.asarray(data.convert("RGB"))
    if isinstance(data, list) and data:               # エクスプローラーでコピーしたファイル
        try:
            with Image.open(data[0]) as im:
                return np.asarray(im.convert("RGB"))
        except OSError:
            return None
    return None


def set_image(img: np.ndarray) -> None:
    """画像を CF_DIB 形式でクリップボードに入れる（Windows のみ）。"""
    if sys.platform != "win32":
        raise OSError("Windows 専用の機能です")
    import win32clipboard
    buf = io.BytesIO()
    Image.fromarray(img).convert("RGB").save(buf, "BMP")
    dib = buf.getvalue()[14:]                         # 先頭14バイトのファイルヘッダを除く
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib)
    finally:
        win32clipboard.CloseClipboard()               # 例外時も必ず閉じる
