"""Quantization utilities (CQ KV-cache integration)."""

from importlib import import_module
from types import ModuleType

from .cq_cache import (
    CQQuantizationConfig,
    CodebookManager,
    QuantizedDynamicCache,
    enable_cq_kv_cache,
)
from .cq_cache_PreRoPE import (
    CQQuantizationConfig as PreRoPECQQuantizationConfig,
)
from .cq_cache_PreRoPE import CodebookManager as PreRoPECodebookManager
from .cq_cache_PreRoPE import (
    QuantizedDynamicCache as PreRoPEQuantizedDynamicCache,
)
from .cq_cache_PreRoPE import (
    enable_cq_kv_cache as enable_prerope_cq_kv_cache,
)

_CQ_BACKEND_MODULES = {
    "postrope": ".cq_cache",
    "post_rope": ".cq_cache",
    "default": ".cq_cache",
    "prerope": ".cq_cache_PreRoPE",
    "pre_rope": ".cq_cache_PreRoPE",
}


def get_cq_backend_module(rope_mode: str = "postrope") -> ModuleType:
    """Return the CQ backend module for the requested RoPE pipeline."""

    normalized_mode = rope_mode.strip().lower()
    try:
        module_name = _CQ_BACKEND_MODULES[normalized_mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_CQ_BACKEND_MODULES))
        raise ValueError(
            f"Unsupported cq_rope_mode={rope_mode!r}. Expected one of: {supported}"
        ) from exc
    return import_module(module_name, package=__name__)


__all__ = [
    "CQQuantizationConfig",
    "CodebookManager",
    "QuantizedDynamicCache",
    "enable_cq_kv_cache",
    "PreRoPECQQuantizationConfig",
    "PreRoPECodebookManager",
    "PreRoPEQuantizedDynamicCache",
    "enable_prerope_cq_kv_cache",
    "get_cq_backend_module",
]
