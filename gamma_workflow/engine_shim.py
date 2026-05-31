"""Thin wrapper over the existing engine module (GeminiRevisionGammaProfile3.py).

We intentionally do NOT refactor the engine. The shim exposes only the functions the
new workflow modules need, and isolates the Colab/Drive cruft. To use this:

    1. Drop GeminiRevisionGammaProfile3.py at the repo root (or anywhere on sys.path).
    2. Import primitives from this module:

        from gamma_workflow.engine_shim import (
            build_refined_options_chain,
            calculate_gex,
            find_flip_level,
            black_scholes_metrics_vectorized,
            calculate_iv_index,
        )

If the engine module is missing, this shim raises a clear error at import time.
"""
from __future__ import annotations
import importlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENGINE_MODULE_NAME = "GeminiRevisionGammaProfile3"


def _load_engine():
    # Try standard import first.
    try:
        return importlib.import_module(_ENGINE_MODULE_NAME)
    except ModuleNotFoundError:
        pass

    # Fallback: look in the repo root.
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / f"{_ENGINE_MODULE_NAME}.py"
    if candidate.exists():
        sys.path.insert(0, str(repo_root))
        return importlib.import_module(_ENGINE_MODULE_NAME)

    raise ImportError(
        f"Could not locate {_ENGINE_MODULE_NAME}.py. Place it at the repo root "
        f"({repo_root}) or install it on PYTHONPATH."
    )


_engine = _load_engine()

# Re-export the primitives we actually use.
build_refined_options_chain = _engine.build_refined_options_chain
calculate_gex = _engine.calculate_gex
find_flip_level = _engine.find_flip_level
black_scholes_metrics_vectorized = _engine.black_scholes_metrics_vectorized
calculate_iv_index = _engine.calculate_iv_index
calculate_price_bounds = _engine.calculate_price_bounds

__all__ = [
    "build_refined_options_chain",
    "calculate_gex",
    "find_flip_level",
    "black_scholes_metrics_vectorized",
    "calculate_iv_index",
    "calculate_price_bounds",
]
