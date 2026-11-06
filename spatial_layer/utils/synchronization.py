"""Thread-safe primitives used by the Layer-2 coordinator.

* :class:`DoubleBuffer` - single-writer / multi-reader double buffer using a
  sequence counter (seqlock). Readers never take a lock and never block the
  writer (e.g. the physics loop or the spatial worker).
* :class:`LatestQueue` - bounded ``queue.Queue`` wrapper that drops the oldest
  item instead of blocking producers (prevents backpressure on vision input).
* :class:`StateHistory` - lock-free ring buffer of robot state snapshots for
  timestamp-based synchronization.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class DoubleBuffer(Generic[T]):
    """Lock-free (for readers) double buffer with an atomic read pointer.

    The writer fills the *back* slot, then bumps the sequence number and flips
    the read index. Under CPython, rebinding an int/list slot is atomic, and the
    sequence counter lets readers detect (and retry on) the rare torn read when
    two writes complete during one read.
    """

    __slots__ = ("_slots", "_read_idx", "_seq", "_write_lock")

    def __init__(self, initial: Optional[T] = None) -> None:
        self._slots: List[Optional[T]] = [initial, initial]
        self._read_idx = 0
        self._seq = 0
        # Only serialises *writers* (normally one); readers never touch it.
        self._write_lock = threading.Lock()

    def write(self, value: T) -> int:
        with self._write_lock:
            back = 1 - self._read_idx
            self._slots[back] = value
            self._seq += 1          # publish ordering: data, seq, pointer
            self._read_idx = back
            return self._seq

    def read(self, max_retries: int = 4) -> Tuple[Optional[T], int]:
        """Return ``(value, sequence)`` without locking."""
        for _ in range(max_retries):
            seq0 = self._seq
            value = self._slots[self._read_idx]
            if self._seq == seq0:
                return value, seq0
        return self._slots[self._read_idx], self._seq

    @property
    def sequence(self) -> int:
        return self._seq


class LatestQueue(Generic[T]):
    """Bounded FIFO that discards the oldest entry when full."""

    def __init__(self, maxsize: int = 2) -> None:
        self._q: "queue.Queue[T]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put(self, item: T) -> bool:
        """Insert ``item``; returns False if a stale frame had to be dropped."""
        dropped = False
        while True:
            try:
                self._q.put_nowait(item)
                return not dropped
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                    dropped = True
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self, timeout: Optional[float] = None) -> Optional[T]:
        """Block for one item, then drain and return only the freshest one."""
        item = self.get(timeout)
        if item is None:
            return None
        while True:
            try:
                newer = self._q.get_nowait()
                self.dropped += 1
                item = newer
            except queue.Empty:
                return item

    def qsize(self) -> int:
        return self._q.qsize()

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class StateHistory(Generic[T]):
    """Fixed-size ring buffer of timestamped snapshots (single writer)."""

    def __init__(self, capacity: int = 64,
                 key: Callable[[Any], float] = lambda s: s.wall_time) -> None:
        self._buf: List[Optional[T]] = [None] * capacity
        self._head = -1          # index of most recent element
        self._count = 0
        self._key = key

    def push(self, item: T) -> None:
        idx = (self._head + 1) % len(self._buf)
        self._buf[idx] = item
        self._head = idx         # atomic publish
        if self._count < len(self._buf):
            self._count += 1

    def latest(self) -> Optional[T]:
        head = self._head
        return None if head < 0 else self._buf[head]

    def closest(self, timestamp: float, max_gap: float = 0.25) -> Optional[T]:
        """Snapshot closest in time to ``timestamp``; falls back to latest."""
        head, count = self._head, self._count
        if head < 0:
            return None
        best, best_dt = None, float("inf")
        n = len(self._buf)
        for i in range(count):
            item = self._buf[(head - i) % n]
            if item is None:
                continue
            dt = abs(self._key(item) - timestamp)
            if dt < best_dt:
                best, best_dt = item, dt
            elif dt > best_dt:
                break            # snapshots are time-ordered: moving away
        if best is None or best_dt > max_gap:
            return self._buf[head]
        return best

    def __len__(self) -> int:
        return self._count

