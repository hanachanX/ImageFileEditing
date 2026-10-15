"""Stable Diffusion (AUTOMATIC1111) が PNG に埋め込む生成情報の読み取り。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class GenInfo:
    prompt: str = ""
    negative: str = ""
    other: str = ""

    def __bool__(self):
        return bool(self.prompt or self.negative or self.other)


_NEG = "\nNegative prompt:"
_STEPS = re.compile(r"(?:^|\n)(Steps: .*)$", re.S)


def parse_parameters(text: str) -> GenInfo:
    """A1111 の 'parameters' 文字列を 3 つに分ける。

    形式:  <prompt>\\nNegative prompt: <negative>\\nSteps: ..., Sampler: ...
    """
    text = text.replace("\r\n", "\n").strip()
    other = ""
    m = _STEPS.search(text)
    if m:
        other = m.group(1).strip()
        text = text[:m.start()].rstrip()
    prompt, negative = text, ""
    if _NEG in "\n" + text:
        body = "\n" + text
        i = body.index(_NEG)
        prompt = body[:i].strip()
        negative = body[i + len(_NEG):].strip()
    return GenInfo(prompt=prompt, negative=negative, other=other)


def _fix_mojibake(text: str) -> str:
    """UTF-8 の文字が tEXt（Latin-1 として読まれる）に入っていた場合に、文字化けを直す。"""
    if all(ord(c) < 128 for c in text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def read_gen_info(path) -> GenInfo:
    """画像ファイルから生成情報を読む。無ければ空の GenInfo。"""
    path = Path(path)
    if path.suffix.lower() != ".png":
        return GenInfo()
    try:
        with Image.open(path) as im:
            text = im.info.get("parameters")
            if not text:
                return GenInfo()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            return parse_parameters(_fix_mojibake(text))
    except (OSError, ValueError):
        return GenInfo()
