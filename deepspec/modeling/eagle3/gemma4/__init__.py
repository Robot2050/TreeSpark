from .config import build_draft_config

try:
    from .modeling import Gemma4Eagle3Model
except ModuleNotFoundError as exc:
    # Gemma4 support requires a transformers build with
    # ``transformers.models.gemma4``.  When it is missing (e.g. older releases),
    # degrade gracefully so the Qwen3 code paths keep working; instantiating a
    # Gemma4 draft then fails with a clear error at build time.
    if not (exc.name or "").startswith("transformers.models.gemma4"):
        raise
    Gemma4Eagle3Model = None

__all__ = [
    "Gemma4Eagle3Model",
    "build_draft_config",
]
