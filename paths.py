"""アプリのフォルダ位置。どこから起動しても同じ場所を指すようにする。"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent      # Image.pyw があるフォルダ
SETTINGS_FILE = APP_DIR / "settings.json"
LEGACY_DIR_CONFIG = APP_DIR / "dir_config.ini"        # 旧版の設定ファイル（読み込みのみ）
PROMPT_FILE = APP_DIR / "prompt.txt"
CARN_WEIGHTS = APP_DIR / "CARN_model.pt"
WEIGHTS_DIR = APP_DIR / "weights"
