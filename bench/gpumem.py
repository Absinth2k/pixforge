"""Sample dedicated GPU memory of a process via Windows PDH (English counter names).

Works on localized Windows because it uses PdhAddEnglishCounterW.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes

pdh = ctypes.WinDLL("pdh")
for _fn in (
    "PdhOpenQueryW",
    "PdhAddEnglishCounterW",
    "PdhCollectQueryData",
    "PdhGetFormattedCounterArrayW",
):
    getattr(pdh, _fn).restype = wintypes.DWORD  # PDH status codes are unsigned

PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2


class _Value(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _Item(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]


class GpuMemSampler:
    """Background sampler of the current process' dedicated GPU memory (bytes)."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._query = wintypes.HANDLE()
        self._counter = wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)):
            raise OSError("PdhOpenQuery failed")
        path = f"\\GPU Process Memory(pid_{os.getpid()}*)\\Dedicated Usage"
        if pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(self._counter)):
            raise OSError("PdhAddEnglishCounter failed")
        pdh.PdhCollectQueryData(self._query)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def read(self) -> int:
        pdh.PdhCollectQueryData(self._query)
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        rc = pdh.PdhGetFormattedCounterArrayW(
            self._counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if rc != PDH_MORE_DATA or size.value == 0:
            return 0
        buf = (ctypes.c_byte * size.value)()
        rc = pdh.PdhGetFormattedCounterArrayW(
            self._counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf
        )
        if rc:
            return 0
        items = ctypes.cast(buf, ctypes.POINTER(_Item))
        return sum(items[i].FmtValue.largeValue for i in range(count.value))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.read())
            time.sleep(self.interval)

    def start(self) -> GpuMemSampler:
        self.baseline = self.read()
        self.peak = self.baseline
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.read())
        return self.peak

    def reset_peak(self) -> None:
        self.peak = self.read()
