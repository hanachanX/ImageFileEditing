"""プロンプトジェネレータの文章づくり（画面から独立）。"""
from __future__ import annotations

from pathlib import Path

EMPHASIS_LEVELS = (0, 1, 2, 3)                      # 括弧の重ね数
WEIGHTS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


def load_prompt_words(path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def format_term(word: str, emphasis: int = 0, weight: float = 1.0) -> str:
    """単語を 'word:1.2' や '((word))' の形にする。

    重みと括弧の両方があるときは A1111 の書式 '(word:1.2)' にする。
    """
    if emphasis not in EMPHASIS_LEVELS:
        raise ValueError("emphasis は 0〜3")
    if weight != 1.0:
        if emphasis == 0:
            return f"{word}:{weight:g}"
        inner = f"{word}:{weight:g}"
    else:
        inner = word
    return "(" * emphasis + inner + ")" * emphasis


class PromptBuilder:
    """単語を ', ' でつないでいく。1つ前の状態に戻せる。"""

    def __init__(self):
        self._terms: list[str] = []

    def add(self, word: str, emphasis: int = 0, weight: float = 1.0) -> str:
        self._terms.append(format_term(word, emphasis, weight))
        return self.text

    def undo(self) -> str:
        if self._terms:
            self._terms.pop()
        return self.text

    def clear(self) -> None:
        self._terms.clear()

    @property
    def text(self) -> str:
        return ", ".join(self._terms)
