from importlib import import_module
from typing import List

"""
tardigrades package initializer.

This module lazily exposes symbols from .core so importing the package
doesn't import the whole implementation until needed.
"""


__all__: List[str]

def _load_core():
    return import_module(".core", __name__)

# Try to discover exports from core without forcing import for attribute access
try:
    _core = _load_core()
    try:
        __all__ = list(getattr(_core, "__all__"))
    except Exception:
        __all__ = [n for n in dir(_core) if not n.startswith("_")]
    __version__ = getattr(_core, "__version__", None)
except Exception:
    # Core may not be importable at package-import time; fall back to lazy resolution
    __all__ = []
    __version__ = None
    _core = None

def __getattr__(name: str):
    """
    Lazily import attributes from tardigrades.core.
    """
    if name == "__version__":
        # Ensure version comes from core if available
        module = _load_core()
        return getattr(module, "__version__", None)
    if name in __all__:
        module = _load_core()
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    names = set(globals().keys())
    names.update(__all__)
    return sorted(names)