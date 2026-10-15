"""設定の読み書き（settings.json）。旧版の dir_config.ini も一度だけ取り込む。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Settings:
    source_dir: str = ""            # 左の一覧のフォルダ
    dest_dir: str = ""              # 右の一覧のフォルダ
    show_info_window: bool = True   # ファイル情報ウインドウを表示
    zoom_on_click: bool = True      # 左クリックで拡大（原寸表示⇔画面に合わせる）
    confirm_move: bool = True       # →キーで移動するとき確認する
    undo_limit: int = 20            # 元に戻せる回数

    def save(self, path: Path) -> None:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)           # 書き込み途中で落ちても壊れないよう、置き換えで保存する


def load_settings(path: Path, legacy_ini: Path | None = None) -> Settings:
    """settings.json を読む。無ければ旧 dir_config.ini（1行目=送り元、2行目=送り先）を取り込む。"""
    s = Settings()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Settings)}
            for k, v in data.items():
                if k in known and isinstance(v, type(getattr(s, k))):
                    setattr(s, k, v)
        except (OSError, ValueError) as e:
            log.warning("設定ファイルを読めませんでした（初期値で起動します）: %s", e)
        return s
    if legacy_ini is not None and legacy_ini.is_file():
        try:
            lines = legacy_ini.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2:
                s.source_dir, s.dest_dir = lines[0].strip(), lines[1].strip()
        except (OSError, UnicodeDecodeError) as e:
            log.warning("旧設定を読めませんでした: %s", e)
    return s
