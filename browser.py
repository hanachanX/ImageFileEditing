"""ファイル一覧ウインドウ（左: 送り元、右: 送り先）と生成情報の表示。"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from .. import files


class FolderPane(ttk.Frame):
    """フォルダ1つ分: パス表示・開くボタン・画像ファイルの一覧。"""

    def __init__(self, master, placeholder: str, on_open: Callable[[], None],
                 on_select: Callable[["FolderPane", str], None],
                 on_delete: Callable[["FolderPane"], None],
                 on_move: Callable[["FolderPane"], None] | None = None,
                 on_context: Callable[["FolderPane", tk.Event], None] | None = None):
        super().__init__(master)
        self.folder: Path | None = None
        self.sort = "name"
        self.names: list[str] = []
        self._on_select, self._on_delete = on_select, on_delete

        top = ttk.Frame(self)
        top.pack(fill="x")
        self.path_var = tk.StringVar(value=placeholder)
        ttk.Entry(top, textvariable=self.path_var, state="readonly", width=28, justify="right").pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(top, text="Open", command=on_open, width=6).pack(side="left")

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, pady=(6, 0))
        self.listbox = tk.Listbox(mid, width=34, height=26, exportselection=False, activestyle="none")
        sb = ttk.Scrollbar(mid, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._selected)
        self.listbox.bind("<Delete>", lambda e: on_delete(self))
        if on_move:
            self.listbox.bind("<Right>", lambda e: on_move(self))
        if on_context:
            self.listbox.bind("<Button-3>", lambda e: on_context(self, e))

    # ---- 一覧
    def set_folder(self, folder) -> None:
        self.folder = Path(folder) if folder else None
        self.path_var.set(str(self.folder) if self.folder else "")
        self.names = []
        self.refresh(force=True)

    def selected_name(self) -> str | None:
        sel = self.listbox.curselection()
        return self.listbox.get(sel[0]) if sel else None

    def selected_path(self) -> Path | None:
        n = self.selected_name()
        return self.folder / n if (self.folder and n) else None

    def refresh(self, force: bool = False) -> bool:
        """フォルダの中身と一覧を同期する。変わったら True。選択は名前で保つ。"""
        new = files.list_images(self.folder, self.sort) if self.folder else []
        if new == self.names and not force:
            return False
        keep = self.selected_name()
        top = self.listbox.yview()[0]
        self.names = new
        self.listbox.delete(0, "end")
        for n in new:
            self.listbox.insert("end", n)
        if keep in new:
            i = new.index(keep)
            self.listbox.selection_set(i)
            self.listbox.activate(i)
        self.listbox.yview_moveto(top)
        return True

    def select_index(self, i: int, notify: bool = True) -> None:
        if not self.names:
            return
        i = max(0, min(i, len(self.names) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.activate(i)
        self.listbox.see(i)
        if notify:
            self._on_select(self, self.names[i])

    def select_name(self, name: str, notify: bool = False) -> None:
        if name in self.names:
            self.select_index(self.names.index(name), notify)

    def _selected(self, _event=None):
        n = self.selected_name()
        if n:
            self._on_select(self, n)


class InfoPanel(ttk.Frame):
    """プロンプト / ネガティブプロンプト / その他の生成情報。"""

    def __init__(self, master):
        super().__init__(master)
        self.boxes: dict[str, tk.Text] = {}
        for row, (key, title, copy) in enumerate((("prompt", "Prompt", True),
                                                  ("negative", "Negative Prompt", True),
                                                  ("other", "Generated Info.", False))):
            head = ttk.Frame(self)
            head.grid(row=row * 2, column=0, sticky="ew", pady=(8, 2))
            ttk.Label(head, text=title, font=("Helvetica", 11, "bold")).pack(side="left")
            if copy:
                ttk.Button(head, text="Copy", width=6, command=lambda k=key: self.copy(k)).pack(side="right")
            box = tk.Text(self, width=44, height=6, wrap="word", state="disabled")
            box.grid(row=row * 2 + 1, column=0, sticky="nsew")
            self.boxes[key] = box
        self.columnconfigure(0, weight=1)

    def show(self, info) -> None:
        for key, box in self.boxes.items():
            box.config(state="normal")
            box.delete("1.0", "end")
            box.insert("end", getattr(info, key) if info else "")
            box.config(state="disabled")

    def copy(self, key: str) -> None:
        text = self.boxes[key].get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
