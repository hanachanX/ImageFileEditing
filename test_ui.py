"""画面の動作確認（仮想ディスプレイ上で実際にウインドウを作って操作する）。

Windows 固有の部分（ゴミ箱、クリップボード、エクスプローラー）は差し替えて確認している。
"""
import os
import time

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

tk = pytest.importorskip("tkinter")
try:
    _r = tk.Tk()
    _r.destroy()
except tk.TclError:
    pytest.skip("画面（DISPLAY）が使えません", allow_module_level=True)

from imagetool import files, operations as O                      # noqa: E402
from imagetool.ui import app as app_mod                           # noqa: E402
from imagetool.ui.tasks import TaskRunner                         # noqa: E402


def pump(app, seconds=0.1):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def wait(app, cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def env(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(), dst.mkdir()
    y, x = np.mgrid[0:120, 0:200]
    base = np.stack([x / 200 * 255, y / 120 * 255, (x + y) / 320 * 255], -1).astype(np.uint8)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "1girl, smile\nNegative prompt: lowres\nSteps: 20, Seed: 1")
    for n in ("a.png", "b.png", "c.png"):
        Image.fromarray(base).save(src / n, pnginfo=meta)
    msgs = []
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: msgs.append(("error", a)))
    monkeypatch.setattr(app_mod.messagebox, "showinfo", lambda *a, **k: msgs.append(("info", a)))
    monkeypatch.setattr(app_mod.messagebox, "askyesno", lambda *a, **k: True)
    # ゴミ箱は Windows 専用なので、テストでは普通の削除に差し替える
    monkeypatch.setattr(files, "delete_to_trash", lambda p: os.remove(p))
    a = app_mod.App(settings_path=tmp_path / "settings.json")
    a.settings.source_dir, a.settings.dest_dir = str(src), str(dst)
    a.left.set_folder(src)
    a.right.set_folder(dst)
    a.update()
    yield a, src, dst, msgs
    try:
        a.destroy()
    except tk.TclError:
        pass


def test_select_loads_image_and_info(env):
    a, src, dst, msgs = env
    a.left.select_index(0, notify=True)
    pump(a)
    assert a.doc.has_image and a.doc.size == (200, 120)
    assert a.info.boxes["prompt"].get("1.0", "end-1c") == "1girl, smile"
    assert a.info.boxes["negative"].get("1.0", "end-1c") == "lowres"
    assert "200x120" in a.status_right.cget("text")


def test_preview_cancel_and_commit_with_undo(env):
    a, src, dst, msgs = env
    a.left.select_index(0, notify=True)
    orig = a.doc.image.copy()
    a.start_operation(O.BY_ID["gamma"])
    dlg = a._op_dialog
    dlg._vars["value"].set(4.0)
    dlg._slider(dlg._op.params[0])
    pump(a, 0.3)
    assert not np.array_equal(a.view._arr, orig), "プレビューが画面に出ていない"
    assert np.array_equal(a.doc.image, orig), "プレビュー中に元の画像が変わってはいけない"
    dlg._close()
    pump(a)
    assert a._op_dialog is None and np.array_equal(a.view._arr, orig)

    a.start_operation(O.BY_ID["gamma"])
    dlg = a._op_dialog
    dlg._vars["value"].set(4.0)
    dlg._ok()
    assert wait(a, lambda: a._op_dialog is None)
    assert a.doc.can_undo and not np.array_equal(a.doc.image, orig)
    a.undo()
    assert np.array_equal(a.doc.image, orig)
    a.redo()
    assert not np.array_equal(a.doc.image, orig)


def test_no_image_gives_message_not_crash(env):
    a, src, dst, msgs = env
    a.start_operation(O.BY_ID["gray"])
    assert msgs and msgs[-1][0] == "info" and not a.runner.busy


def test_instant_operation_and_geometry(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    a.start_operation(O.BY_ID["rot_r"])
    assert wait(a, lambda: a.doc.size == (120, 200))
    a.undo()
    assert a.doc.size == (200, 120)


def test_crop_with_selection(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    a.crop_selection()                                   # 選択なし → 案内だけ
    assert a.doc.size == (200, 120)
    a.view._sel = (10, 20, 60, 70)
    a.crop_selection()
    assert a.doc.size == (50, 50)


def test_mosaic_uses_selection(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    orig = a.doc.image.copy()
    a.view._sel = (0, 0, 40, 40)
    a.start_operation(O.BY_ID["mosaic"])
    a._op_dialog._ok()
    assert wait(a, lambda: a._op_dialog is None)
    assert np.array_equal(a.doc.image[60:], orig[60:])
    assert not np.array_equal(a.doc.image[:40, :40], orig[:40, :40])


def test_delete_selects_next_and_loads_it(env):
    a, src, dst, msgs = env
    a.left.select_index(0, notify=True)
    a.left.listbox.focus_force()
    a.delete_selected(a.left)
    assert not (src / "a.png").exists()
    assert a.left.names == ["b.png", "c.png"] and a.left.selected_name() == "b.png"
    assert a.doc.path == src / "b.png"
    a.left.select_index(1, notify=True)                  # 最後の1つを消すと、1つ前が選ばれる
    a.delete_selected(a.left)
    assert a.left.selected_name() == "b.png"
    a.delete_selected(a.left)
    assert a.left.names == [] and not a.doc.has_image


def test_move_to_right(env):
    a, src, dst, msgs = env
    a.settings.confirm_move = False
    a.left.select_index(1, notify=True)
    a.move_selected(a.left)
    assert (dst / "b.png").exists() and not (src / "b.png").exists()
    assert a.right.names == ["b.png"] and a.left.names == ["a.png", "c.png"]
    assert a.left.selected_name() == "c.png"
    Image.new("RGB", (4, 4)).save(src / "b.png")          # 同名があるときは上書きせずエラー
    a.left.refresh()
    a.left.select_name("b.png")
    a.move_selected(a.left)
    assert msgs[-1][0] == "error" and (src / "b.png").exists()


def test_move_without_destination(env):
    a, src, dst, msgs = env
    a.right.set_folder(None)
    a.left.select_index(0, notify=True)
    a.move_selected(a.left)
    assert msgs[-1][0] == "error" and (src / "a.png").exists()


def test_folder_polling_sees_outside_changes(env):
    """旧版は『エクスプローラーで消したファイルが一覧に残る』既知の不具合があった。"""
    a, src, dst, msgs = env
    a.left.select_index(1, notify=True)
    os.remove(src / "a.png")
    Image.new("RGB", (4, 4)).save(src / "z.png")
    a.refresh_folders()
    assert a.left.names == ["b.png", "c.png", "z.png"]
    assert a.left.selected_name() == "b.png"              # 選択は名前で保たれる


def test_broken_image_does_not_crash(env):
    a, src, dst, msgs = env
    a.left.select_index(0, notify=True)
    before = a.doc.image.copy()
    (src / "bad.png").write_bytes(b"garbage")
    assert a.load_path(src / "bad.png") is False
    assert msgs[-1][0] == "error" and np.array_equal(a.doc.image, before)


def test_rename(env, monkeypatch):
    a, src, dst, msgs = env
    import tkinter.simpledialog as sd
    monkeypatch.setattr(sd, "askstring", lambda *x, **k: "renamed")
    a.left.select_index(0, notify=True)
    a.rename_selected(a.left)
    assert (src / "renamed.png").exists() and a.doc.path == src / "renamed.png"
    assert a.left.selected_name() == "renamed.png"


def test_paste_empty_clipboard(env, monkeypatch):
    a, *_ = env
    monkeypatch.setattr(app_mod.clipboard, "get_image", lambda: None)
    a.paste_image()
    assert msgs_last(env)[0] == "info" and not a.doc.has_image


def msgs_last(env):
    return env[3][-1]


def test_paste_image(env, monkeypatch):
    a, *_ = env
    monkeypatch.setattr(app_mod.clipboard, "get_image", lambda: np.zeros((30, 40, 3), np.uint8))
    a.paste_image()
    assert a.doc.size == (40, 30) and a.doc.path is None


def test_copy_failure_is_reported(env, monkeypatch):
    a, *_ = env
    a.left.select_index(0, notify=True)

    def boom(_):
        raise OSError("no clipboard")
    monkeypatch.setattr(app_mod.clipboard, "set_image", boom)
    a.copy_image()
    assert env[3][-1][0] == "error"


def test_upscale_without_torch_reports_error(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    before = a.doc.image.copy()
    a._run_upscale("carn", 1.5, 4)
    assert wait(a, lambda: not a.runner.busy)
    pump(a, 0.2)
    import importlib.util
    if importlib.util.find_spec("torch"):
        pytest.skip("torch がある環境では失敗しない")
    assert env[3][-1][0] == "error" and np.array_equal(a.doc.image, before)


def test_ascii_and_promptgen_windows_open(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    a.open_ascii()
    a.open_promptgen()
    a.open_settings()
    pump(a)
    assert all(w.winfo_exists() for w in a._children.values())
    a.open_ascii()                                       # 二重に開かない
    assert len(a._children) == 3


def test_settings_toggle_persists(env):
    a, src, dst, msgs = env
    a.hide_browser()
    assert not a.browser.winfo_viewable()
    assert not app_mod.load_settings(a._settings_path).show_info_window
    a.show_browser()
    pump(a)
    assert a.browser.winfo_viewable()


def test_click_toggles_fit(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    assert a.view.fit_mode
    a._on_canvas_click()
    assert not a.view.fit_mode
    a.settings.zoom_on_click = False
    a._on_canvas_click()
    assert not a.view.fit_mode


# ---------------------------------------------------------------- スレッド実行
def test_task_runner_done_error_progress(env):
    a, *_ = env
    runner = TaskRunner(a)
    got = {}
    runner.run(lambda p: (p(1, 2), p(2, 2), 42)[2], lambda r: got.update(done=r),
               lambda e: got.update(err=e), lambda d, t: got.setdefault("prog", []).append((d, t)))
    assert wait(a, lambda: "done" in got) and got["done"] == 42 and got["prog"] == [(1, 2), (2, 2)]
    runner.run(lambda p: 1 / 0, lambda r: got.update(done2=r), lambda e: got.update(err=e))
    assert wait(a, lambda: "err" in got) and isinstance(got["err"], ZeroDivisionError)
    assert wait(a, lambda: not runner.busy)


def test_busy_blocks_second_operation(env):
    a, *_ = env
    a.left.select_index(0, notify=True)
    import threading
    gate = threading.Event()
    a.runner.run(lambda p: gate.wait(2), lambda r: None, lambda e: None)
    a.start_operation(O.BY_ID["gray"])
    assert a._op_dialog is None and not a.doc.can_undo   # 実行中は新しい処理を始めない
    gate.set()
    assert wait(a, lambda: not a.runner.busy)


def test_quit_saves_settings(env):
    a, src, dst, msgs = env
    a.quit_app()
    s = app_mod.load_settings(a._settings_path)
    assert s.source_dir == str(src) and s.dest_dir == str(dst)
