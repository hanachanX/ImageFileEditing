"""小さなダイアログ類。処理の登録表(operations)から自動で作る汎用ダイアログを含む。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .. import promptgen
from ..operations import Operation, Param


def quantize(p: Param, v: float) -> float:
    """スライダーの値を、刻み幅にそろえて範囲内に収める。"""
    if p.step:
        v = p.lo + round((v - p.lo) / p.step) * p.step
    v = min(max(v, p.lo), p.hi)
    return int(round(v)) if p.kind == "int" else round(v, 6)


class OperationDialog(tk.Toplevel):
    """パラメータを動かしながらプレビューし、OK で確定するダイアログ。

    compute(values) -> 結果の画像（失敗時は例外）、preview(arr|None)、commit(values) は呼び出し側が渡す。
    """

    def __init__(self, master, op: Operation, preview: Callable, commit: Callable, cancel: Callable,
                 busy_flag: Callable[[], bool] = lambda: False):
        super().__init__(master)
        self.title(op.label)
        self.transient(master)
        self.resizable(False, False)
        self._op, self._preview, self._commit, self._cancel = op, preview, commit, cancel
        self._vars: dict[str, tk.Variable] = {}
        self._labels: dict[str, ttk.Label] = {}
        self._job = None
        self._done = False

        body = ttk.Frame(self, padding=12)
        body.grid(sticky="nsew")
        for row, p in enumerate(op.params):
            ttk.Label(body, text=p.label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)
            if p.kind == "choice":
                var = tk.IntVar(value=int(p.default))
                self._vars[p.key] = var
                box = ttk.Frame(body)
                box.grid(row=row, column=1, columnspan=2, sticky="w")
                for text, value in p.choices:
                    ttk.Radiobutton(box, text=text, value=value, variable=var,
                                    command=self._changed).pack(anchor="w")
            else:
                var = tk.DoubleVar(value=p.default)
                self._vars[p.key] = var
                sc = ttk.Scale(body, from_=p.lo, to=p.hi, variable=var, length=220,
                               command=lambda _v, p=p: self._slider(p))
                sc.grid(row=row, column=1, pady=6)
                lab = ttk.Label(body, width=8, anchor="e")
                lab.grid(row=row, column=2, padx=(8, 0))
                self._labels[p.key] = lab
                sc.bind("<MouseWheel>", lambda e, p=p, v=var: self._wheel(p, v, e.delta))
                sc.bind("<Button-4>", lambda e, p=p, v=var: self._wheel(p, v, 1))
                sc.bind("<Button-5>", lambda e, p=p, v=var: self._wheel(p, v, -1))
                self._show_value(p)
        self.error = ttk.Label(body, foreground="#b00020", wraplength=300)
        self.error.grid(row=len(op.params), column=0, columnspan=3, sticky="w")
        btns = ttk.Frame(body)
        btns.grid(row=len(op.params) + 1, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.ok_btn = ttk.Button(btns, text="OK", command=self._ok)
        self.ok_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="キャンセル", command=self._close).pack(side="left")
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)
        if op.live:
            self.after(10, self._changed)

    # ---- 値
    def values(self) -> dict:
        out = {}
        for p in self._op.params:
            v = self._vars[p.key].get()
            out[p.key] = int(v) if p.kind == "choice" else quantize(p, v)
        return out

    def _show_value(self, p: Param):
        v = quantize(p, self._vars[p.key].get())
        self._labels[p.key].config(text=p.fmt.format(v))

    def _slider(self, p: Param):
        self._show_value(p)
        self._changed()

    def _wheel(self, p: Param, var, delta):
        step = p.step or 1
        var.set(quantize(p, var.get() + (step if delta > 0 else -step)))
        self._slider(p)
        return "break"

    def _changed(self):
        if not self._op.live:
            return
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(60, self._do_preview)           # 連続操作は間引く

    def _do_preview(self):
        self._job = None
        self.error.config(text="")
        try:
            self._preview(self.values())
        except Exception as e:                                  # noqa: BLE001 - 画面に出すだけ
            self.error.config(text=str(e))

    # ---- 確定・取消
    def set_busy(self, busy: bool):
        self.ok_btn.config(state="disabled" if busy else "normal")

    def show_error(self, text: str):
        self.error.config(text=text)

    def _ok(self):
        if self._done:
            return
        self._commit(self.values(), self)

    def finish(self):
        self._done = True
        self.destroy()

    def _close(self):
        if not self._done:
            self._done = True
            self._cancel()
        self.destroy()


class UpscaleDialog(tk.Toplevel):
    def __init__(self, master, run: Callable[[str, float, int], None]):
        super().__init__(master)
        self.title("超解像（拡大）")
        self.transient(master)
        self.resizable(False, False)
        self._run = run
        f = ttk.Frame(self, padding=12)
        f.grid()
        self.model = tk.StringVar(value="carn")
        self.native = tk.IntVar(value=4)
        self.factor = tk.DoubleVar(value=2.0)
        ttk.Radiobutton(f, text="CARN_V2（軽い・最大2倍）", value="carn", variable=self.model,
                        command=self._sync).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(f, text="Real-ESRGAN（きれい・重い・最大4倍）", value="esrgan", variable=self.model,
                        command=self._sync).grid(row=1, column=0, columnspan=3, sticky="w")
        self.nat_box = ttk.Frame(f)
        self.nat_box.grid(row=2, column=0, columnspan=3, sticky="w", padx=(24, 0))
        ttk.Label(self.nat_box, text="使うモデル:").pack(side="left")
        ttk.Radiobutton(self.nat_box, text="x4", value=4, variable=self.native).pack(side="left", padx=4)
        ttk.Radiobutton(self.nat_box, text="x2", value=2, variable=self.native).pack(side="left")
        ttk.Label(f, text="倍率").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.scale = ttk.Scale(f, variable=self.factor, length=200, command=lambda v: self._label())
        self.scale.grid(row=3, column=1, pady=(10, 0))
        self.lab = ttk.Label(f, width=6)
        self.lab.grid(row=3, column=2, pady=(10, 0))
        row = ttk.Frame(f)
        row.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(row, text="実行", command=self._go).pack(side="left", padx=4)
        ttk.Button(row, text="閉じる", command=self.destroy).pack(side="left")
        self._sync()

    def _sync(self):
        from ..upscale import engine
        lo, hi = engine.CARN_RANGE if self.model.get() == "carn" else engine.ESRGAN_RANGE
        self.scale.config(from_=lo, to=hi)
        self.factor.set(min(max(self.factor.get(), lo), hi))
        for child in self.nat_box.winfo_children():
            child.state(["!disabled"] if self.model.get() == "esrgan" else ["disabled"])
        self._label()

    def _label(self):
        self.lab.config(text=f"{round(self.factor.get(), 1):.1f}x")

    def _go(self):
        self._run(self.model.get(), round(self.factor.get(), 1), self.native.get())
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, master, settings, on_change: Callable[[], None]):
        super().__init__(master)
        self.title("設定")
        self.transient(master)
        self.resizable(False, False)
        f = ttk.Frame(self, padding=14)
        f.grid()
        self.v_info = tk.BooleanVar(value=settings.show_info_window)
        self.v_zoom = tk.BooleanVar(value=settings.zoom_on_click)
        self.v_conf = tk.BooleanVar(value=settings.confirm_move)

        def push():
            settings.show_info_window = self.v_info.get()
            settings.zoom_on_click = self.v_zoom.get()
            settings.confirm_move = self.v_conf.get()
            on_change()

        for r, (text, var) in enumerate((
                ("ファイル情報ウインドウを表示する", self.v_info),
                ("画像をクリックで「画面に合わせる／原寸」を切り替える", self.v_zoom),
                ("→キーでファイルを移動するとき確認する", self.v_conf))):
            ttk.Checkbutton(f, text=text, variable=var, command=push).grid(row=r, sticky="w", pady=4)
        ttk.Button(f, text="閉じる", command=self.destroy).grid(row=3, sticky="e", pady=(10, 0))


class AsciiWindow(tk.Toplevel):
    """アスキーアート表示。文字づくりは imageops.ascii_art が行う。"""

    def __init__(self, master, make: Callable[[int], str]):
        super().__init__(master)
        from ..imageops import ascii_art
        self.title("ASCII ART")
        self._make = make
        self._text = ""
        self._invert = tk.BooleanVar(value=True)
        bar = ttk.Frame(self, padding=6)
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="幅(文字数)").pack(side="left")
        self.cols = tk.IntVar(value=80)
        ttk.Scale(bar, from_=ascii_art.MIN_COLUMNS, to=ascii_art.MAX_COLUMNS, variable=self.cols,
                  length=160).pack(side="left", padx=6)
        self.lab = ttk.Label(bar, width=4)
        self.lab.pack(side="left")
        ttk.Button(bar, text="作成", command=self.generate).pack(side="left", padx=6)
        ttk.Checkbutton(bar, text="白黒反転", variable=self._invert, command=self._paint).pack(side="left")
        self.cols.trace_add("write", lambda *_: self.lab.config(text=str(int(self.cols.get()))))
        self.lab.config(text="80")
        self.view = tk.Text(self, font=("Courier New", 4), wrap="none", width=100, height=50, borderwidth=0)
        self.view.pack(fill="both", expand=True)
        self.view.config(state="disabled")
        self.generate()

    def generate(self):
        try:
            self._text = self._make(int(self.cols.get()))
        except Exception as e:                                   # noqa: BLE001
            self._text = f"作成できませんでした: {e}\n"
        self._paint()

    def _paint(self):
        bg, fg = ("black", "white") if self._invert.get() else ("white", "black")
        self.view.config(state="normal", background=bg, foreground=fg)
        self.view.delete("1.0", "end")
        self.view.insert("end", self._text)
        self.view.config(state="disabled")


class PromptGenWindow(tk.Toplevel):
    def __init__(self, master, words: list[str]):
        super().__init__(master)
        self.title("Prompt Generator")
        self.geometry("500x760")
        self.builder = promptgen.PromptBuilder()
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=10, pady=10)
        self.listbox = tk.Listbox(top, height=18, exportselection=False)
        sb = ttk.Scrollbar(top, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        for w in words:
            self.listbox.insert("end", w)
        if not words:
            self.listbox.insert("end", "(prompt.txt が空か、見つかりません)")

        opt = ttk.Frame(self)
        opt.pack(fill="x", padx=10)
        self.emph = tk.IntVar(value=0)
        self.weight = tk.DoubleVar(value=1.0)
        ttk.Label(opt, text="強調").grid(row=0, column=0, sticky="w")
        for i, (t, v) in enumerate((("なし", 0), ("( )", 1), ("(( ))", 2), ("((( )))", 3))):
            ttk.Radiobutton(opt, text=t, value=v, variable=self.emph).grid(row=0, column=i + 1, padx=4)
        ttk.Label(opt, text="重み").grid(row=1, column=0, sticky="w", pady=6)
        for i, w in enumerate(promptgen.WEIGHTS):
            ttk.Radiobutton(opt, text=f"{w:g}", value=w, variable=self.weight).grid(
                row=1 + i // 4, column=1 + i % 4, padx=4, sticky="w")
        self.text = tk.Text(self, height=8, wrap="word")
        self.text.pack(fill="x", padx=10, pady=10)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(row, text="追加（ダブルクリックでも可）", command=self.add).pack(side="left")
        ttk.Button(row, text="1つ戻す", command=self.undo).pack(side="left", padx=6)
        ttk.Button(row, text="クリア", command=self.clear).pack(side="left")
        ttk.Button(row, text="コピー", command=self.copy).pack(side="right")
        self.listbox.bind("<Double-Button-1>", lambda e: self.add())

    def _refresh(self):
        self.text.delete("1.0", "end")
        self.text.insert("end", self.builder.text)

    def add(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.builder.add(self.listbox.get(sel[0]), self.emph.get(), round(self.weight.get(), 1))
        self._refresh()

    def undo(self):
        self.builder.undo()
        self._refresh()

    def clear(self):
        self.builder.clear()
        self._refresh()

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.text.get("1.0", "end-1c"))




__all__ = ["OperationDialog", "UpscaleDialog", "SettingsDialog", "AsciiWindow", "PromptGenWindow", "quantize"]
