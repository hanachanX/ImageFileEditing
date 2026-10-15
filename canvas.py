"""画像の表示、拡大縮小（画面に合わせる／原寸）、ドラッグでの範囲選択。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from ..imageops.geometry import normalize_box


class ImageCanvas(ttk.Frame):
    def __init__(self, master, on_click=None, on_context=None, on_redraw=None):
        super().__init__(master)
        self._on_redraw = on_redraw
        self._on_click = on_click
        self._on_context = on_context
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#2b2b2b")
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._arr: np.ndarray | None = None
        self._photo = None
        self._fit = True                    # True: 画面に合わせる / False: 原寸
        self._scale = 1.0
        self._sel: tuple[int, int, int, int] | None = None   # 画像座標での選択範囲
        self._drag_start = None
        self._redraw_job = None

        self.canvas.bind("<Configure>", lambda e: self._schedule_redraw())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._context)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))   # X11
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

    # ---- 表示
    def set_image(self, arr: np.ndarray | None, keep_selection: bool = False) -> None:
        self._arr = arr
        if not keep_selection:
            self._sel = None
        self._redraw()

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def fit_mode(self) -> bool:
        return self._fit

    def toggle_fit(self) -> None:
        self._fit = not self._fit
        self._redraw()

    def _compute_scale(self) -> float:
        if self._arr is None:
            return 1.0
        if not self._fit:
            return 1.0
        h, w = self._arr.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        return min(1.0, cw / w, ch / h)

    def _schedule_redraw(self):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(40, self._redraw)

    def _show_bars(self, show: bool):
        """「画面に合わせる」ときはスクロールバーが要らないので隠す。"""
        if show:
            self.vbar.grid()
            self.hbar.grid()
        else:
            self.vbar.grid_remove()
            self.hbar.grid_remove()

    def _redraw(self):
        self._redraw_job = None
        self._show_bars(not self._fit)
        self.canvas.delete("all")
        if self._arr is None:
            self._photo = None
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            if self._on_redraw:
                self._on_redraw()
            return
        h, w = self._arr.shape[:2]
        self._scale = s = self._compute_scale()
        im = Image.fromarray(self._arr)
        if s != 1.0:
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(im)          # 参照を保持しないと消える
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw", tags="image")
        self.canvas.configure(scrollregion=(0, 0, im.width, im.height))
        self._draw_selection()
        if self._on_redraw:
            self._on_redraw()

    # ---- 選択範囲
    @property
    def selection(self):
        return self._sel

    def clear_selection(self) -> None:
        self._sel = None
        self.canvas.delete("sel")

    def _draw_selection(self):
        self.canvas.delete("sel")
        if self._sel:
            s = self._scale
            x0, y0, x1, y1 = (v * s for v in self._sel)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffffff", width=1, tags="sel")
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#000000", dash=(4, 4), width=1, tags="sel")

    def _to_canvas(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _press(self, event):
        self.canvas.focus_set()
        self._drag_start = self._to_canvas(event)

    def _motion(self, event):
        if self._drag_start is None or self._arr is None:
            return
        x, y = self._to_canvas(event)
        sx, sy = self._drag_start
        self.canvas.delete("sel")
        self.canvas.create_rectangle(sx, sy, x, y, outline="#ffffff", dash=(4, 4), tags="sel")

    def _release(self, event):
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        x, y = self._to_canvas(event)
        self._drag_start = None
        if self._arr is None:
            return
        if abs(x - sx) < 4 and abs(y - sy) < 4:        # ほぼ動かない = クリック
            self.clear_selection()
            if self._on_click:
                self._on_click()
            return
        s = self._scale
        h, w = self._arr.shape[:2]
        self._sel = normalize_box((sx / s, sy / s, x / s, y / s), w, h)
        self._draw_selection()

    def _context(self, event):
        if self._on_context:
            self._on_context(event)
