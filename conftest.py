import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def rgb():
    """ランダムな RGB 画像（37x53 ＝ 8でも14でも割り切れない大きさ）。"""
    return np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)


@pytest.fixture
def rgb_aligned():
    """56x112 ＝ 8でも14でも割り切れる大きさ。旧版と完全一致を比べるため。"""
    return np.random.default_rng(1).integers(0, 256, (56, 112, 3), dtype=np.uint8)
