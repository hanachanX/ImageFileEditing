"""画像処理の登録表。

メニューとパラメータ入力ダイアログはこの表から自動で作られるので、
新しい処理を足すときは Operation を1つ追加するだけでよい。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .imageops import adjust, effects, filters, geometry


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str = "float"                  # "int" | "float" | "choice"
    lo: float = 0
    hi: float = 1
    default: float = 0
    step: float = 1
    choices: tuple = ()                  # kind="choice" のとき ((表示名, 値), ...)
    fmt: str = "{:g}"                    # 値の表示書式


@dataclass(frozen=True)
class Operation:
    id: str
    label: str
    group: str                           # メニューの小見出し
    func: Callable
    params: tuple = ()
    uses_selection: bool = False         # 範囲選択があれば region= として渡す
    live: bool = True                    # 動かしながらプレビューする（重い処理は False）


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


OPERATIONS: tuple[Operation, ...] = (
    # --- 色・明るさ
    Operation("rgb", "RGB バランス", "色・明るさ", lambda img, r, g, b: adjust.rgb_gain(img, r, g, b), (
        Param("r", "R", lo=0, hi=3, default=1, step=0.1, fmt="{:.1f}"),
        Param("g", "G", lo=0, hi=3, default=1, step=0.1, fmt="{:.1f}"),
        Param("b", "B", lo=0, hi=3, default=1, step=0.1, fmt="{:.1f}"))),
    Operation("hsv", "色相・彩度・明度", "色・明るさ",
              lambda img, h, s, v: adjust.hsv_adjust(img, h, s, v), (
        Param("h", "色相ずらし(度)", "int", 0, 360, 0, 1),
        Param("s", "彩度の倍率", lo=0, hi=2, default=1, step=0.05, fmt="{:.2f}"),
        Param("v", "明度の加減", "int", -255, 255, 0, 1))),
    Operation("contrast", "コントラスト", "色・明るさ", lambda img, alpha: adjust.contrast(img, alpha), (
        Param("alpha", "倍率", lo=0.4, hi=1.6, default=1, step=0.01, fmt="{:.2f}"),)),
    Operation("brightness", "明るさ", "色・明るさ", lambda img, delta: adjust.brightness(img, delta), (
        Param("delta", "加減", "int", -100, 100, 0, 1),)),
    Operation("gamma", "ガンマ補正", "色・明るさ", lambda img, value: adjust.gamma(img, value), (
        Param("value", "ガンマ値", lo=0.1, hi=5.0, default=2.2, step=0.1, fmt="{:.1f}"),)),
    Operation("gray", "グレースケール", "色・明るさ", adjust.grayscale),
    Operation("sepia", "セピア", "色・明るさ", adjust.sepia),
    Operation("posterize", "ポスタリゼーション", "色・明るさ", lambda img, levels: adjust.posterize(img, levels), (
        Param("levels", "階調数", "int", 2, 4, 2, 1),)),
    # --- フィルタ
    Operation("blur", "ぼかし (Gaussian)", "フィルタ",
              lambda img, kernel, sigma: filters.gaussian_blur(img, _odd(kernel), sigma), (
        Param("kernel", "カーネル(奇数)", "int", 1, 31, 5, 2),
        Param("sigma", "シグマ", lo=0, hi=10, default=1, step=0.1, fmt="{:.1f}"))),
    Operation("sharpen", "シャープ", "フィルタ", lambda img, amount: filters.sharpen(img, amount), (
        Param("amount", "強さ", lo=0, hi=2, default=0.5, step=0.05, fmt="{:.2f}"),)),
    Operation("dilate", "膨張", "フィルタ", filters.dilate),
    Operation("erode", "収縮", "フィルタ", filters.erode),
    Operation("clahe", "CLAHE (局所コントラスト)", "フィルタ", filters.clahe, live=False),
    Operation("pencil", "鉛筆画", "フィルタ", filters.pencil_sketch, live=False),
    # --- 効果
    Operation("mosaic", "モザイク", "効果", lambda img, block, region=None: effects.mosaic(img, block, region), (
        Param("block", "ブロックの大きさ", "int", 1, 60, 10, 1),), uses_selection=True),
    Operation("halftone", "ハーフトーン", "効果", lambda img, method: effects.halftone(img, int(method)), (
        Param("method", "方式", "choice", default=1,
              choices=tuple((v, k) for k, v in effects.HALFTONE_METHODS.items())),), live=False),
    Operation("dot", "ドットアート", "効果", lambda img, size: effects.dot_art(img, size), (
        Param("size", "ドットの大きさ", "int", 3, 40, 10, 1),), live=False),
    Operation("painterly", "絵画風", "効果", lambda img: effects.painterly(img), live=False),
    # --- 変形
    Operation("resize", "拡大縮小", "変形", lambda img, factor: geometry.resize(img, factor), (
        Param("factor", "倍率", lo=0.1, hi=3.0, default=1.1, step=0.1, fmt="{:.1f}x"),), live=False),
    Operation("flip_h", "左右反転", "変形", geometry.flip_horizontal),
    Operation("flip_v", "上下反転", "変形", geometry.flip_vertical),
    Operation("rot_r", "右に90度回転", "変形", geometry.rotate_right),
    Operation("rot_l", "左に90度回転", "変形", geometry.rotate_left),
)

BY_ID = {op.id: op for op in OPERATIONS}
GROUPS = tuple(dict.fromkeys(op.group for op in OPERATIONS))


def default_values(op: Operation) -> dict:
    out = {}
    for p in op.params:
        out[p.key] = int(p.default) if p.kind in ("int", "choice") else p.default
    return out


def run(op: Operation, img, values: dict | None = None, region=None):
    """処理を実行して新しい画像を返す（元の配列は書き換えない）。"""
    kwargs = default_values(op)
    if values:
        kwargs.update(values)
    if op.uses_selection:
        kwargs["region"] = region
    return op.func(img, **kwargs)
