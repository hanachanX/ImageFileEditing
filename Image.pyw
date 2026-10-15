"""IMAGE FILE TOOL 起動用ファイル（ダブルクリックで起動）。"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from imagetool.ui.app import main
    main()
except Exception:                                  # pythonw ではエラーが見えないので、ファイルに残す
    Path(__file__).with_name("image_tool_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    raise
