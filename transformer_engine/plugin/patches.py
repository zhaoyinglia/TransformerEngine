"""FL device patches invoked by the upstream NVTE_PLUGIN patch hook."""

from importlib import import_module


def apply_patches():
    """Apply available vendor patches before PyTorch submodules are imported."""
    for vendor in ("musa", "tsingmicro", "npu"):
        try:
            import_module(
                f"transformer_engine.plugin.core.backends.vendor.{vendor}.patches"
            ).apply_patch()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
