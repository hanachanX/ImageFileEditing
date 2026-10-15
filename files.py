"""フォルダ内の画像一覧と、ファイルの移動・改名・削除・ZIP化。"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder, sort: str = "name") -> list[str]:
    """画像ファイル名の一覧。sort は 'name'（名前順）か 'time'（更新日時順）。"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    names = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if sort == "time":
        return sorted(names, key=lambda n: ((folder / n).stat().st_mtime, n.lower()))
    return sorted(names, key=str.lower)


def move_file(src_dir, name: str, dest_dir) -> Path:
    """ファイルを別フォルダへ移す。同名があれば FileExistsError。"""
    src, dest = Path(src_dir) / name, Path(dest_dir) / name
    if dest.exists():
        raise FileExistsError(name)
    shutil.move(str(src), str(dest))
    return dest


def rename_file(folder, old: str, new: str) -> Path:
    new = new.strip()
    if not new or any(c in new for c in '\\/:*?"<>|') or new in (".", ".."):
        raise ValueError("ファイル名に使えない文字が入っています")
    if Path(new).suffix.lower() != Path(old).suffix.lower():
        new += Path(old).suffix                       # 拡張子を消しても元のものを保つ
    src, dst = Path(folder) / old, Path(folder) / new
    if dst.exists() and dst != src:
        raise FileExistsError(new)
    src.rename(dst)
    return dst


def delete_to_trash(path) -> None:
    """ゴミ箱へ送る（Windows）。ゴミ箱が使えない環境では例外にし、勝手に完全削除はしない。"""
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if sys.platform != "win32":
        raise OSError("この環境ではゴミ箱に送れません")
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT = 3, 0x40, 0x10, 0x4
    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    op.pFrom = path + "\0"                            # 二重 NUL 終端（LPCWSTR が最後の NUL を足す）
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0 or op.fAnyOperationsAborted:
        raise OSError(f"ゴミ箱へ送れませんでした (code {rc})")


def archive_images(folder, names=None) -> Path:
    """フォルダの画像を <フォルダ名>.zip にまとめる（元ファイルは消さない）。"""
    folder = Path(folder)
    names = list(names) if names is not None else list_images(folder)
    if not names:
        raise ValueError("圧縮する画像がありません")
    out = folder / (folder.name + ".zip")
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(folder / n, arcname=n)
    tmp.replace(out)
    return out


def reveal_in_explorer(path) -> None:
    """エクスプローラーでファイル（またはフォルダ）を開く。"""
    import subprocess
    p = Path(path)
    if sys.platform == "win32":
        if p.is_dir():
            os.startfile(str(p))                      # noqa: S606
        else:
            subprocess.Popen(["explorer", f"/select,{os.path.normpath(p)}"])
    else:
        raise OSError("Windows 専用の機能です")
