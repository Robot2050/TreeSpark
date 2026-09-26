import os

import torch


def _npu_module():
    module = getattr(torch, "npu", None)
    if module is None:
        return None
    try:
        return module if module.is_available() else None
    except Exception:
        return None


def is_npu_available() -> bool:
    requested = os.environ.get("DEEPSPEC_DEVICE", "").strip().lower()
    if requested == "cuda":
        return False
    if requested == "npu":
        return _npu_module() is not None
    return _npu_module() is not None


def device_type() -> str:
    requested = os.environ.get("DEEPSPEC_DEVICE", "").strip().lower()
    if requested in {"cuda", "npu"}:
        return requested
    if _npu_module() is not None:
        return "npu"
    return "cuda"


def accelerator_module():
    return getattr(torch, device_type())


def accelerator_backend() -> str:
    return "hccl" if device_type() == "npu" else "nccl"


def device_count() -> int:
    return int(accelerator_module().device_count())


def set_device(local_rank: int) -> None:
    accelerator_module().set_device(int(local_rank))


def current_device_index() -> int:
    return int(accelerator_module().current_device())


def make_device(local_rank: int | None = None) -> torch.device:
    if local_rank is None:
        local_rank = current_device_index()
    return torch.device(device_type(), int(local_rank))


def manual_seed_all(seed: int) -> None:
    if device_count() > 0:
        accelerator_module().manual_seed_all(int(seed))


def empty_cache() -> None:
    module = accelerator_module()
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def synchronize_device(device: torch.device) -> None:
    """Synchronize ``device`` if its accelerator runtime is available.

    No-op on CPU-only builds and on devices whose runtime is missing, so it is
    safe to call before wall-clock timing points on any platform.
    """
    module = getattr(torch, str(device.type), None)
    synchronize = getattr(module, "synchronize", None)
    is_available = getattr(module, "is_available", None)
    if synchronize is None or is_available is None or not is_available():
        return
    module.synchronize(device)


def get_rng_state():
    return accelerator_module().get_rng_state()


def set_rng_state(state) -> None:
    accelerator_module().set_rng_state(state)
