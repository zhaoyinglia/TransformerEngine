# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

def __getattr__(name):
    if name in __all__:
        from importlib import import_module

        core = import_module(".core", __name__)
        if name == "tefl":
            return core.get_tefl_module()
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TEFLBackendBase",
    "TEFLModule",
    "get_tefl_module",
    "get_registry",
    "tefl",
]


def load_plugins():
    """NVTE_PLUGIN entrypoint: expose FL dispatch under the public extension name."""
    import sys

    from .core import get_tefl_module

    sys.modules["transformer_engine_torch"] = get_tefl_module()
