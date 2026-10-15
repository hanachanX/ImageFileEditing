"""重い処理を別スレッドで実行し、結果は必ず画面側のスレッドで受け取る。

Tk の部品は別スレッドから触ってはいけないので、スレッドは Queue に結果を入れるだけにして、
画面側が after() で定期的に取り出す。
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from typing import Callable


class TaskRunner:
    def __init__(self, widget: tk.Misc, interval_ms: int = 50):
        self._widget = widget
        self._q: queue.Queue = queue.Queue()
        self._interval = interval_ms
        self._busy = 0
        self._polling = False

    @property
    def busy(self) -> bool:
        return self._busy > 0

    def run(self, work: Callable[[Callable[[int, int], None]], object],
            on_done: Callable[[object], None],
            on_error: Callable[[BaseException], None],
            on_progress: Callable[[int, int], None] | None = None) -> None:
        """work(progress) を別スレッドで実行する。progress(done, total) で進み具合を伝えられる。"""
        self._busy += 1

        def progress(done, total):
            self._q.put(("progress", (done, total), on_progress))

        def target():
            try:
                result = work(progress)
            except BaseException as e:                  # noqa: BLE001 - 画面側へ渡す
                self._q.put(("error", e, on_error))
            else:
                self._q.put(("done", result, on_done))

        threading.Thread(target=target, daemon=True).start()
        if not self._polling:
            self._polling = True
            self._widget.after(self._interval, self._poll)

    def _poll(self):
        try:
            while True:
                kind, payload, cb = self._q.get_nowait()
                if kind == "progress":
                    if cb:
                        cb(*payload)
                    continue
                self._busy -= 1
                cb(payload)
        except queue.Empty:
            pass
        except tk.TclError:                             # 画面が閉じられた
            self._polling = False
            return
        if self._busy > 0:
            try:
                self._widget.after(self._interval, self._poll)
            except tk.TclError:
                self._polling = False
        else:
            self._polling = False
