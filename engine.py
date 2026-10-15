"""CARN_V2 / Real-ESRGAN で拡大する。モデルは一度読んだら使い回す。

torch / torchvision はこのファイルの関数の中で初めて読み込む。
（インストールされていなくても、アプリ本体は起動できる。）
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import paths

ProgressFn = Callable[[int, int], None]

CARN_RANGE = (1.1, 2.0)        # CARN_V2 は2倍モデルを縮めて使うので、2.0倍まで
ESRGAN_RANGE = (1.1, 4.0)

_lock = threading.Lock()
_cache: dict = {}


class UpscaleUnavailable(RuntimeError):
    """torch が無い、重みファイルが無い、などで実行できないとき。"""


def _torch():
    try:
        import torch
    except ImportError as e:                           # pragma: no cover - 環境依存
        raise UpscaleUnavailable(
            "超解像には PyTorch が必要です。requirements-upscale.txt を参照してインストールしてください。") from e
    return torch


def _device(torch):
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _final_resize(arr: np.ndarray, factor: float, native: float) -> np.ndarray:
    """モデルの出力(native倍)を、指定倍率(factor)にそろえる。"""
    if abs(factor - native) < 1e-6:
        return arr
    ratio = factor / native
    return cv2.resize(arr, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)


def load_carn(weights: Path | None = None):
    weights = Path(weights or paths.CARN_WEIGHTS)
    torch = _torch()
    dev = _device(torch)
    key = ("carn", str(dev), str(weights))
    with _lock:
        if key in _cache:
            return _cache[key], dev
        if not weights.is_file():
            raise UpscaleUnavailable(f"重みファイルがありません: {weights}")
        from .carn_model import CARN_V2, network_to_half
        net = CARN_V2(color_channels=3, mid_channels=64, conv=torch.nn.Conv2d,
                      single_conv_size=3, single_conv_group=1, scale=2,
                      activation=torch.nn.LeakyReLU(0.1), SEBlock=True,
                      repeat_blocks=3, atrous=(1, 1, 1))
        net = network_to_half(net)
        net.load_state_dict(torch.load(weights, map_location=dev))
        net = net.to(dev) if dev.type == "cuda" else net.float()
        net.eval()
        _cache[key] = net
        return net, dev


def upscale_carn(img: np.ndarray, factor: float) -> np.ndarray:
    """CARN_V2 で拡大。factor は 1.1〜2.0。"""
    if not (CARN_RANGE[0] <= factor <= CARN_RANGE[1]):
        raise ValueError(f"倍率は{CARN_RANGE[0]}〜{CARN_RANGE[1]}にしてください")
    torch = _torch()
    net, dev = load_carn()
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = net(x)
    out = out.float().squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    out = _final_resize(out, factor, 2.0)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def load_esrgan(native_scale: int, progress: ProgressFn | None = None):
    if native_scale not in (2, 4):
        raise ValueError("Real-ESRGAN は 2 倍か 4 倍モデルを選びます")
    torch = _torch()
    dev = _device(torch)
    weights = paths.WEIGHTS_DIR / f"RealESRGAN_x{native_scale}.pth"
    key = ("esrgan", native_scale, str(dev))
    with _lock:
        if key in _cache:
            return _cache[key]
        from .realesrgan.model import RealESRGAN
        paths.WEIGHTS_DIR.mkdir(exist_ok=True)
        model = RealESRGAN(dev, scale=native_scale)
        try:
            model.load_weights(str(weights), download=True)   # 無ければ自動ダウンロード
        except FileNotFoundError as e:
            raise UpscaleUnavailable(f"重みファイルがありません: {weights}") from e
        _cache[key] = model
        return model


def upscale_esrgan(img: np.ndarray, factor: float, native_scale: int = 4,
                   progress: ProgressFn | None = None) -> np.ndarray:
    """Real-ESRGAN で拡大。native_scale=2 or 4 のモデルを使い、factor 倍にそろえる。"""
    if not (ESRGAN_RANGE[0] <= factor <= ESRGAN_RANGE[1]):
        raise ValueError(f"倍率は{ESRGAN_RANGE[0]}〜{ESRGAN_RANGE[1]}にしてください")
    model = load_esrgan(native_scale)
    out = np.asarray(model.predict(img, progress=progress))
    return _final_resize(out, factor, float(native_scale))
