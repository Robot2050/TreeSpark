import contextlib
import threading

import torch
import torch.distributed as dist

from .distributed import is_global_main_process, print_on_global_main

try:
    import hfai

    HAS_HFAI = True
except ModuleNotFoundError:
    hfai = None
    HAS_HFAI = False


class SuspendController:
    def __init__(self, device, poll_interval_seconds: float = 1.0):
        self.device = device
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._requested = threading.Event()
        self._stop_requested = threading.Event()
        self._monitor_thread = None
        self._suspend_flag = torch.zeros(1, device=self.device, dtype=torch.int32)

    def _monitor_suspend(self):
        print_on_global_main("Start monitoring hfai suspend signal...")
        while not self._stop_requested.is_set():
            if hfai.client.receive_suspend_command():
                print_on_global_main("Received hfai suspend command!")
                self._requested.set()
                return
            self._stop_requested.wait(timeout=self.poll_interval_seconds)

    @contextlib.contextmanager
    def monitoring(self):
        if not HAS_HFAI:
            yield self
            return

        try:
            self._requested.clear()
            self._stop_requested.clear()
            if is_global_main_process():
                self._monitor_thread = threading.Thread(
                    target=self._monitor_suspend,
                    daemon=True,
                )
                self._monitor_thread.start()
            yield self
        finally:
            self._stop_requested.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=self.poll_interval_seconds + 1.0)
                self._monitor_thread = None

    def requested(self) -> bool:
        if not HAS_HFAI:
            return False

        dist.barrier()
        if is_global_main_process():
            self._suspend_flag[0] = 1 if self._requested.is_set() else 0
        dist.broadcast(self._suspend_flag, src=0)
        return bool(self._suspend_flag.item())

    def go_suspend(self):
        if not HAS_HFAI:
            raise RuntimeError("hfai is not available; cannot suspend the job.")
        hfai.client.go_suspend()
