"""メインウインドウ（画像表示と処理）と、ファイル一覧ウインドウをつなぐ司令塔。

ここには「どの部品をいつ呼ぶか」だけを書き、画像処理・ファイル操作・設定の中身は別モジュールにある。
"""
from __future__ import annotations

import datetime
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import __version__, clipboard, files, metadata, operations, paths, promptgen
from ..document import Document
from ..imageops import ascii_art, geometry
from ..settings import load_settings
from . import dialogs
from .browser import FolderPane, InfoPanel
from .canvas import ImageCanvas
from .tasks import TaskRunner

log = logging.getLogger(__name__)
POLL_MS = 1500          # フォルダの変化を見に行く間隔


class App(tk.Tk):
    def __init__(self, settings_path: Path | None = None):
        super().__init__()
        self.title(f"IMAGE FILE TOOL {__version__}")
        self.geometry("900x700+100+60")
        self._settings_path = Path(settings_path or paths.SETTINGS_FILE)
        self.settings = load_settings(self._settings_path, paths.LEGACY_DIR_CONFIG)
        self.doc = Document(self.settings.undo_limit)
        self.runner = TaskRunner(self)
        self._op_dialog: dialogs.OperationDialog | None = None
        self._preview_arr = None
        self._children: dict[str, tk.Toplevel] = {}

        self._build_main()
        self._build_menus()
        self._build_browser()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

        if self.settings.source_dir:
            self.left.set_folder(self.settings.source_dir)
        if self.settings.dest_dir:
            self.right.set_folder(self.settings.dest_dir)
        self._apply_settings()
        self.after(POLL_MS, self._poll_folders)

    # ================================================================ 画面の組み立て
    def _build_main(self):
        self.view = ImageCanvas(self, on_click=self._on_canvas_click, on_context=self._show_edit_menu,
                                on_redraw=self._refresh_status)
        self.view.pack(fill="both", expand=True)
        bar = ttk.Frame(self, relief="sunken", padding=(8, 2))
        bar.pack(side="bottom", fill="x")
        self.status_left = ttk.Label(bar, anchor="w")
        self.status_right = ttk.Label(bar, anchor="e")
        self.status_left.pack(side="left", fill="x", expand=True)
        self.status_right.pack(side="right")

    def _build_menus(self):
        bar = tk.Menu(self)
        m = tk.Menu(bar, tearoff=0)
        m.add_command(label="開く...  Ctrl+O", command=self.open_file)
        m.add_command(label="名前を付けて保存...  Ctrl+S", command=self.save_file)
        m.add_separator()
        m.add_command(label="ファイル一覧ウインドウを表示", command=self.show_browser)
        m.add_separator()
        m.add_command(label="終了", command=self.quit_app)
        bar.add_cascade(label="ファイル", menu=m)

        self.edit_menu = tk.Menu(bar, tearoff=0)
        for item in (("元に戻す  Ctrl+Z", self.undo), ("やり直し  Ctrl+Y", self.redo), None,
                           ("コピー  Ctrl+C", self.copy_image), ("貼り付け  Ctrl+V", self.paste_image), None,
                           ("選択範囲で切り抜き", self.crop_selection), ("選択範囲を解除  Esc", self.view.clear_selection)):
            if item is None:
                self.edit_menu.add_separator()
            else:
                self.edit_menu.add_command(label=item[0], command=item[1])
        bar.add_cascade(label="編集", menu=self.edit_menu)

        conv = tk.Menu(bar, tearoff=0)
        for group in operations.GROUPS:
            sub = tk.Menu(conv, tearoff=0)
            for op in operations.OPERATIONS:
                if op.group == group:
                    sub.add_command(label=op.label, command=lambda op=op: self.start_operation(op))
            conv.add_cascade(label=group, menu=sub)
        bar.add_cascade(label="変換", menu=conv)

        sr = tk.Menu(bar, tearoff=0)
        sr.add_command(label="拡大 (CARN / Real-ESRGAN)...", command=self.open_upscale)
        bar.add_cascade(label="超解像", menu=sr)

        bar.add_command(label="アスキーアート", command=self.open_ascii)
        bar.add_command(label="プロンプト作成", command=self.open_promptgen)
        bar.add_command(label="設定", command=self.open_settings)
        self.config(menu=bar)

    def _build_browser(self):
        win = tk.Toplevel(self)
        win.title("File Info")
        win.geometry("760x900+1010+60")
        win.protocol("WM_DELETE_WINDOW", self.hide_browser)
        self.browser = win
        panes = ttk.Frame(win, padding=8)
        panes.pack(fill="both", expand=True)
        self.left = FolderPane(panes, "送り元フォルダ", lambda: self.choose_folder("source"), self._on_pick,
                               self.delete_selected, self.move_selected, self._pane_menu)
        self.right = FolderPane(panes, "送り先フォルダ", lambda: self.choose_folder("dest"), self._on_pick,
                                self.delete_selected, None, self._pane_menu)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right.grid(row=0, column=1, sticky="nsew")
        panes.columnconfigure((0, 1), weight=1)
        panes.rowconfigure(0, weight=3)
        self.info = InfoPanel(panes)
        self.info.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        panes.rowconfigure(1, weight=2)

        menu = tk.Menu(win)
        cmd = tk.Menu(menu, tearoff=0)
        cmd.add_command(label="エクスプローラーで開く（左）", command=lambda: self.reveal(self.left))
        cmd.add_command(label="エクスプローラーで開く（右）", command=lambda: self.reveal(self.right))
        cmd.add_command(label="名前を変更（左）", command=lambda: self.rename_selected(self.left))
        cmd.add_command(label="名前を変更（右）", command=lambda: self.rename_selected(self.right))
        menu.add_cascade(label="コマンド", menu=cmd)
        menu.add_command(label="ZIPにまとめる（左）", command=self.archive_left)
        menu.add_command(label="右を名前順", command=lambda: self.sort_right("name"))
        menu.add_command(label="右を日時順", command=lambda: self.sort_right("time"))
        win.config(menu=menu)
        self.pane_menu = tk.Menu(win, tearoff=0)
        self._ctx_pane: FolderPane | None = None
        self.pane_menu.add_command(label="エクスプローラーで開く", command=lambda: self.reveal(self._ctx_pane))
        self.pane_menu.add_command(label="名前を変更", command=lambda: self.rename_selected(self._ctx_pane))
        self.pane_menu.add_command(label="削除（ゴミ箱へ）", command=lambda: self.delete_selected(self._ctx_pane))

    def _bind_keys(self):
        for seq, fn in (("<Control-o>", self.open_file), ("<Control-s>", self.save_file),
                        ("<Control-z>", self.undo), ("<Control-y>", self.redo),
                        ("<Control-c>", self.copy_image), ("<Control-v>", self.paste_image),
                        ("<Escape>", self.view.clear_selection)):
            self.bind(seq, lambda e, fn=fn: fn())

    # ================================================================ 表示まわり
    def _show_doc(self, fit_window: bool = False, keep_selection: bool = False):
        self.view.set_image(self.doc.image, keep_selection=keep_selection)
        if fit_window:
            self._fit_window()
        self._refresh_status()

    def _fit_window(self):
        if not self.doc.has_image:
            return
        w, h = self.doc.size
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        tw = max(420, min(w + 20, int(sw * 0.62)))
        th = max(320, min(h + 60, int(sh * 0.85)))
        self.geometry(f"{tw}x{th}")

    def _refresh_status(self):
        if not self.doc.has_image:
            self.status_left.config(text="")
            self.status_right.config(text="")
            return
        w, h = self.doc.size
        self.status_right.config(text=f"{w}x{h}  {self.view.scale * 100:.0f}%")
        if self.doc.path and self.doc.path.exists():
            t = datetime.datetime.fromtimestamp(self.doc.path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.status_left.config(text=f"{self.doc.path} : {t}")
        else:
            self.status_left.config(text="(貼り付け／未保存)")

    def _on_canvas_click(self):
        if self.settings.zoom_on_click and self.doc.has_image and self._op_dialog is None:
            self.view.toggle_fit()

    def _show_edit_menu(self, event):
        self.edit_menu.post(event.x_root, event.y_root)

    def _notify(self, text: str):
        self.status_left.config(text=text)

    def _error(self, text: str):
        messagebox.showerror("エラー", text, parent=self)

    def _need_image(self) -> bool:
        if not self.doc.has_image:
            messagebox.showinfo("画像がありません", "先に画像を開いてください。", parent=self)
            return False
        return True

    # ================================================================ 読み込み・保存
    def load_path(self, path) -> bool:
        if self._op_dialog is not None:
            self._op_dialog._close()
        try:
            self.doc.load_file(path)
        except Exception as e:                                   # noqa: BLE001 - 壊れた画像などで落とさない
            self._error(f"画像を開けませんでした。\n{path}\n\n{e}")
            return False
        self.info.show(metadata.read_gen_info(path))
        self._show_doc(fit_window=True)
        self.lift()
        return True

    def open_file(self):
        p = filedialog.askopenfilename(
            parent=self, filetypes=[("画像", "*.jpg *.jpeg *.png"), ("すべて", "*.*")],
            initialdir=self.settings.source_dir or None)
        if p:
            self.load_path(p)

    def save_file(self):
        if not self._need_image():
            return
        initial = self.doc.path.name if self.doc.path else ""
        p = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png", initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp")])
        if not p:
            return
        try:
            self.doc.save(p)
        except Exception as e:                                   # noqa: BLE001
            self._error(f"保存できませんでした。\n{e}")
            return
        self._notify(f"保存しました: {p}")
        self.refresh_folders()

    def paste_image(self):
        arr = clipboard.get_image()
        if arr is None:
            messagebox.showinfo("貼り付け", "クリップボードに画像がありません。", parent=self)
            return
        self.doc.load_array(arr)
        self.info.show(None)
        self._show_doc(fit_window=True)

    def copy_image(self):
        if not self._need_image():
            return
        try:
            clipboard.set_image(self.doc.image)
        except Exception as e:                                   # noqa: BLE001
            self._error(f"クリップボードにコピーできませんでした。\n{e}")

    # ================================================================ 編集（元に戻す等）
    def undo(self):
        if self._op_dialog is None and self.doc.undo():
            self._show_doc(fit_window=True)

    def redo(self):
        if self._op_dialog is None and self.doc.redo():
            self._show_doc(fit_window=True)

    def crop_selection(self):
        if not self._need_image():
            return
        sel = self.view.selection
        if not sel:
            messagebox.showinfo("切り抜き", "先に画像の上をドラッグして範囲を選んでください。", parent=self)
            return
        self.doc.apply(geometry.crop(self.doc.image, sel))
        self._show_doc(fit_window=True)

    # ================================================================ 画像処理
    def start_operation(self, op: operations.Operation):
        if not self._need_image():
            return
        if self.runner.busy:
            self._notify("別の処理を実行中です。終わるまでお待ちください。")
            return
        if self._op_dialog is not None:
            self._op_dialog.lift()
            return
        if not op.params:
            self._execute(op, {}, None)
            return
        self._op_dialog = dialogs.OperationDialog(
            self, op,
            preview=lambda values, op=op: self._preview(op, values),
            commit=lambda values, dlg, op=op: self._execute(op, values, dlg),
            cancel=self._cancel_preview)
        self._place_near(self._op_dialog)

    def _place_near(self, win: tk.Toplevel):
        win.update_idletasks()
        n = len([w for w in self._children.values() if w.winfo_exists()])      # 重ならないよう少しずつずらす
        win.geometry(f"+{self.winfo_rootx() + 30 + 40 * n}+{self.winfo_rooty() + 30 + 40 * n}")

    def _compute(self, op, values):
        return operations.run(op, self.doc.image, values, self.view.selection)

    def _preview(self, op, values):
        self._preview_arr = self._compute(op, values)
        self.view.set_image(self._preview_arr, keep_selection=True)

    def _cancel_preview(self):
        self._op_dialog = None
        self._preview_arr = None
        self._show_doc(keep_selection=True)

    def _finish_dialog(self, dlg):
        self._op_dialog = None
        self._preview_arr = None
        if dlg is not None:
            dlg.finish()

    def _execute(self, op, values, dlg):
        """処理を確定する。重い処理は別スレッドで行う。"""
        if dlg is not None:
            dlg.set_busy(True)
        self._notify(f"{op.label} を実行中...")
        self.config(cursor="watch")

        def done(out):
            self.config(cursor="")
            self.doc.apply(out)
            self._finish_dialog(dlg)
            self._show_doc(fit_window=True)

        def failed(err):
            self.config(cursor="")
            if dlg is not None:
                dlg.set_busy(False)
                dlg.show_error(str(err))
            else:
                self._error(f"{op.label} に失敗しました。\n{err}")
            self._show_doc(keep_selection=True)

        # 画面を止めないよう、すべて別スレッドで計算する（読み込んだ配列は書き換えない）
        img, sel = self.doc.image, self.view.selection
        self.runner.run(lambda progress: operations.run(op, img, values, sel), done, failed)

    # ================================================================ 超解像
    def open_upscale(self):
        if not self._need_image():
            return
        dialogs.UpscaleDialog(self, self._run_upscale)

    def _run_upscale(self, model: str, factor: float, native: int):
        from ..upscale import engine
        img = self.doc.image
        label = "CARN" if model == "carn" else "Real-ESRGAN"
        self._notify(f"{label} で拡大中... （初回はモデルの読み込みで時間がかかります）")
        self.config(cursor="watch")

        def work(progress):
            if model == "carn":
                return engine.upscale_carn(img, factor)
            return engine.upscale_esrgan(img, factor, native, progress)

        def done(out):
            self.config(cursor="")
            self.doc.apply(out)
            self._show_doc(fit_window=True)

        def failed(err):
            self.config(cursor="")
            self._show_doc(keep_selection=True)
            self._error(f"超解像に失敗しました。\n{err}")

        self.runner.run(work, done, failed,
                        on_progress=lambda d, t: self._notify(f"{label} で拡大中... {d}/{t}"))

    # ================================================================ 別ウインドウ
    def _single(self, key: str, factory):
        win = self._children.get(key)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return win
        win = factory()
        self._children[key] = win
        self._place_near(win)
        return win

    def open_ascii(self):
        if not self._need_image():
            return
        self._single("ascii", lambda: dialogs.AsciiWindow(
            self, lambda cols: ascii_art.to_ascii(self.doc.image, cols)))

    def open_promptgen(self):
        self._single("prompt", lambda: dialogs.PromptGenWindow(self, promptgen.load_prompt_words(paths.PROMPT_FILE)))

    def open_settings(self):
        self._single("settings", lambda: dialogs.SettingsDialog(self, self.settings, self._on_settings_changed))

    def _on_settings_changed(self):
        self._apply_settings()
        self._save_settings()

    def _apply_settings(self):
        if self.settings.show_info_window:
            self.browser.deiconify()
        else:
            self.browser.withdraw()

    def show_browser(self):
        self.settings.show_info_window = True
        self._on_settings_changed()
        self.browser.lift()

    def hide_browser(self):
        self.settings.show_info_window = False
        self._on_settings_changed()

    def _save_settings(self):
        try:
            self.settings.save(self._settings_path)
        except OSError as e:
            log.warning("設定を保存できませんでした: %s", e)

    # ================================================================ ファイル一覧
    def choose_folder(self, which: str):
        current = self.settings.source_dir if which == "source" else self.settings.dest_dir
        d = filedialog.askdirectory(parent=self.browser, initialdir=current or None)
        if not d:
            return
        if which == "source":
            self.settings.source_dir = d
            self.left.set_folder(d)
        else:
            self.settings.dest_dir = d
            self.right.set_folder(d)
        self._save_settings()

    def _on_pick(self, pane: FolderPane, name: str):
        other = self.right if pane is self.left else self.left
        if pane.folder:
            self.load_path(pane.folder / name)
        other.listbox.selection_clear(0, "end")

    def _pane_menu(self, pane: FolderPane, event):
        i = pane.listbox.nearest(event.y)
        if 0 <= i < len(pane.names):
            pane.select_index(i, notify=False)
        self._ctx_pane = pane
        self.pane_menu.post(event.x_root, event.y_root)

    def refresh_folders(self):
        self.left.refresh()
        self.right.refresh()

    def _poll_folders(self):
        try:
            self.refresh_folders()
        finally:
            self.after(POLL_MS, self._poll_folders)

    def _select_after_removal(self, pane: FolderPane, index: int):
        pane.refresh(force=True)
        if pane.names:
            pane.select_index(index, notify=True)
        else:
            self.doc = Document(self.settings.undo_limit)
            self._show_doc()
            self.info.show(None)

    def delete_selected(self, pane: FolderPane | None):
        if pane is None:
            return
        name = pane.selected_name()
        if not name:
            return
        index = pane.listbox.curselection()[0]
        try:
            files.delete_to_trash(pane.folder / name)
        except Exception as e:                                   # noqa: BLE001
            self._error(f"ゴミ箱へ送れませんでした。\n{e}")
            return
        self._select_after_removal(pane, index)
        pane.listbox.focus_set()

    def move_selected(self, pane: FolderPane):
        name = pane.selected_name()
        if not name:
            return
        if not self.right.folder:
            self._error("先に送り先フォルダを開いてください。")
            return
        if self.settings.confirm_move and not messagebox.askyesno(
                "確認", f"「{name}」を右のフォルダへ移動しますか？", parent=self.browser):
            return
        index = pane.listbox.curselection()[0]
        try:
            files.move_file(pane.folder, name, self.right.folder)
        except FileExistsError:
            self._error("送り先に同じ名前のファイルがあります。")
            return
        except OSError as e:
            self._error(f"移動できませんでした。\n{e}")
            return
        self.right.refresh(force=True)
        self._select_after_removal(pane, index)
        pane.listbox.focus_set()

    def rename_selected(self, pane: FolderPane | None):
        from tkinter import simpledialog
        if pane is None or not pane.selected_name():
            return
        old = pane.selected_name()
        new = simpledialog.askstring("名前の変更", "新しいファイル名:", initialvalue=old, parent=self.browser)
        if not new or new == old:
            return
        try:
            dst = files.rename_file(pane.folder, old, new)
        except FileExistsError:
            self._error("同じ名前のファイルがあります。")
            return
        except (ValueError, OSError) as e:
            self._error(f"名前を変更できませんでした。\n{e}")
            return
        pane.refresh(force=True)
        pane.select_name(dst.name)
        if self.doc.path and self.doc.path == pane.folder / old:
            self.doc.path = dst

    def reveal(self, pane: FolderPane | None):
        if pane is None or not pane.folder:
            self._error("先にフォルダを開いてください。")
            return
        try:
            files.reveal_in_explorer(pane.selected_path() or pane.folder)
        except OSError as e:
            self._error(str(e))

    def archive_left(self):
        if not self.left.folder or not self.left.names:
            self._error("左の一覧に画像がありません。")
            return
        if not messagebox.askyesno("確認", "左の一覧の画像をZIPにまとめます。よろしいですか？\n（元のファイルは消しません）",
                                   parent=self.browser):
            return
        try:
            out = files.archive_images(self.left.folder, self.left.names)
        except Exception as e:                                   # noqa: BLE001
            self._error(f"ZIPを作れませんでした。\n{e}")
            return
        mb = out.stat().st_size / 1024 ** 2
        messagebox.showinfo("完了", f"{out.name} を作りました（{mb:.2f} MB）。\n不要になった元の画像は、ご自身で削除してください。",
                            parent=self.browser)
        self.left.refresh()

    def sort_right(self, mode: str):
        self.right.sort = mode
        self.right.refresh(force=True)

    # ================================================================ 終了
    def quit_app(self):
        self._save_settings()
        self.destroy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    App().mainloop()


__all__ = ["App", "main", "__version__"]
