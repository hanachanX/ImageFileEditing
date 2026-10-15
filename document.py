"""編集中の1枚の画像と、元に戻す／やり直しの履歴。画面（Tk）には依存しない。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .imageops.common import ensure_rgb_uint8


class Document:
    def __init__(self, undo_limit: int = 20):
        self._img: np.ndarray | None = None
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self.undo_limit = max(1, undo_limit)
        self.path: Path | None = None               # 開いたファイル（貼り付けなら None）

    # ---- 状態
    @property
    def image(self) -> np.ndarray | None:
        return self._img

    @property
    def has_image(self) -> bool:
        return self._img is not None

    @property
    def size(self) -> tuple[int, int]:
        """(幅, 高さ)"""
        if self._img is None:
            return (0, 0)
        return (self._img.shape[1], self._img.shape[0])

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ---- 読み込み
    def load_array(self, img: np.ndarray, path=None) -> None:
        """新しい画像として読み込む（履歴は消える）。"""
        self._img = ensure_rgb_uint8(img)
        self._undo.clear()
        self._redo.clear()
        self.path = Path(path) if path else None

    def load_file(self, path) -> None:
        with Image.open(path) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB"))
        self.load_array(rgb, path)

    # ---- 編集
    def apply(self, new_img: np.ndarray) -> None:
        """処理結果を確定する。直前の状態は元に戻せる。"""
        if self._img is None:
            raise ValueError("画像がありません")
        self._undo.append(self._img)
        if len(self._undo) > self.undo_limit:
            del self._undo[0]
        self._redo.clear()
        self._img = ensure_rgb_uint8(new_img)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._img)
        self._img = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._img)
        self._img = self._redo.pop()
        return True

    # ---- 保存
    def save(self, path, jpeg_quality: int = 95) -> None:
        if self._img is None:
            raise ValueError("保存する画像がありません")
        path = Path(path)
        ext = path.suffix.lower()
        im = Image.fromarray(self._img)
        if ext in (".jpg", ".jpeg"):
            im.save(path, quality=jpeg_quality, subsampling=0)
        elif ext in (".png", ".bmp"):
            im.save(path)
        else:
            raise ValueError(f"保存できない形式です: {ext or '(拡張子なし)'}")
