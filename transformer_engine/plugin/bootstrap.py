"""FL initialization around the unmodified upstream common loader."""

import importlib
import os
from pathlib import Path
import runpy


def skip_cuda_build() -> bool:
    """Use the runtime override, then the generated build configuration."""
    if os.environ.get("TE_FL_SKIP_CUDA"):
        return bool(int(os.environ["TE_FL_SKIP_CUDA"]))
    config = Path(__file__).parent / "core/_build_config.py"
    return runpy.run_path(str(config))["SKIP_CUDA_BUILD"] if config.is_file() else False


def import_common():
    """Import upstream common without loading CUDA libraries in FL-only mode."""
    if not skip_cuda_build():
        return importlib.import_module("transformer_engine.common")

    # Upstream's build guard covers import-time checks and native library loads.
    keys = ("NVTE_PROJECT_BUILDING", "NVTE_RELEASE_BUILD")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(NVTE_PROJECT_BUILDING="1", NVTE_RELEASE_BUILD="0")
    try:
        return importlib.import_module("transformer_engine.common")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_pytorch_extension():
    """Select FL-only initialization or the upstream native/plugin sequence."""
    plugin = os.environ.setdefault("NVTE_PLUGIN", "transformer_engine.plugin")
    if skip_cuda_build():
        module = importlib.import_module(plugin)
        module.load_plugins()
        if hasattr(module, "__path__") and importlib.util.find_spec(f"{plugin}.patches") is not None:
            importlib.import_module(f"{plugin}.patches").apply_patches()
    else:
        from transformer_engine.common import load_framework_extension

        load_framework_extension("torch")
