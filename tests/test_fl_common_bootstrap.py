"""CPU regressions for FL common initialization, without loading CUDA libraries."""

import ast
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def common(monkeypatch):
    monkeypatch.setenv("NVTE_PROJECT_BUILDING", "1")
    monkeypatch.setenv("NVTE_RELEASE_BUILD", "0")
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "0")
    monkeypatch.delenv("NVTE_PLUGIN", raising=False)
    # Isolate early FL registration from real hardware discovery.
    parent = ModuleType("transformer_engine")
    parent.__path__ = []
    plugin = ModuleType("transformer_engine.plugin")
    plugin.__path__ = []
    core = ModuleType("transformer_engine.plugin.core")
    parent.plugin = plugin
    plugin.core = core
    for module in (parent, plugin, core):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[1] / "transformer_engine/common/__init__.py"
    spec = importlib.util.spec_from_file_location("_fl_common_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    aliases = ("transformer_engine_torch", "transformer_engine_torch_nv")
    previous = {name: sys.modules.pop(name) for name in aliases if name in sys.modules}
    monkeypatch.setattr(module, "get_te_core_package_info", lambda: (False, "", ""))
    monkeypatch.setattr(module, "_is_package_installed", lambda name: name == "transformer_engine")
    monkeypatch.setattr(module, "_is_package_installed_from_wheel", lambda name: True)
    try:
        yield module
    finally:
        for name in aliases:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


@pytest.fixture
def loader(common, monkeypatch):
    state = SimpleNamespace(error=None, version=91300, loads=0, native=None)
    parent = ModuleType("transformer_engine")
    parent.__path__ = []
    plugin = ModuleType("transformer_engine.plugin")
    plugin.__path__ = []
    core = ModuleType("transformer_engine.plugin.core")
    stub = ModuleType("transformer_engine_torch")
    core.get_tefl_module = lambda: stub
    parent.plugin = plugin
    plugin.core = core
    for name, module in ((parent.__name__, parent), (plugin.__name__, plugin), (core.__name__, core)):
        monkeypatch.setitem(sys.modules, name, module)

    class NativeLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            state.loads += 1
            if state.error:
                raise state.error
            module.get_cudnn_version = lambda: state.version
            state.native = module

    monkeypatch.setattr(common, "_get_shared_object_file", lambda framework: Path("native.so"))
    monkeypatch.setattr(common.importlib.util, "spec_from_file_location", lambda name, path: ModuleSpec(name, NativeLoader()))
    sys.modules["transformer_engine_torch"] = stub
    state.stub = stub
    state.core = core
    return state


def test_native_and_fl_aliases_are_stable(common, loader):
    common.load_framework_extension("torch_nv")
    assert sys.modules["transformer_engine_torch"] is loader.stub
    assert sys.modules["transformer_engine_torch_nv"] is loader.native
    common.load_framework_extension("torch_nv")
    assert loader.loads == 1


@pytest.mark.parametrize("failure", ["exec", "cudnn"])
def test_failed_initialization_can_retry(common, loader, failure):
    if failure == "exec":
        loader.error = ImportError("broken native load")
    elif failure == "cudnn":
        loader.version = 91100
    with pytest.raises((ImportError, RuntimeError)):
        common.load_framework_extension("torch_nv")
    assert sys.modules["transformer_engine_torch"] is loader.stub
    assert "transformer_engine_torch_nv" not in sys.modules
    loader.error = None
    loader.version = 91300
    common.load_framework_extension("torch_nv")
    assert sys.modules["transformer_engine_torch"] is loader.stub


@pytest.mark.parametrize("failure", [False, True])
def test_external_plugin_retains_native_alias(common, loader, monkeypatch, failure):
    external = ModuleType("fl_test_external")
    replacement = ModuleType("external_extension")
    def load_plugins():
        assert sys.modules["transformer_engine_torch_nv"] is loader.native
        sys.modules["transformer_engine_torch"] = replacement
        if failure:
            sys.modules.pop("transformer_engine_torch_nv")
            raise RuntimeError("external setup failed")
    external.load_plugins = load_plugins
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setenv("NVTE_PLUGIN", external.__name__)
    if failure:
        with pytest.warns(RuntimeWarning, match="external setup failed"):
            common.load_framework_extension("torch_nv")
    else:
        common.load_framework_extension("torch_nv")
    assert sys.modules["transformer_engine_torch_nv"] is loader.native
    assert sys.modules["transformer_engine_torch"] is (loader.stub if failure else replacement)


def test_empty_override_uses_build_metadata(common, monkeypatch):
    config = ModuleType("transformer_engine.plugin.core._build_config")
    config.SKIP_CUDA_BUILD = True
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "")
    assert common.skip_cuda_build() is True
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "0")
    assert common.skip_cuda_build() is False


def test_no_cuda_skips_native_and_install_checks(common, loader, monkeypatch):
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
    def forbidden():
        pytest.fail("no-CUDA build queried split CUDA packages")
    monkeypatch.setattr(common, "get_te_core_package_info", forbidden)
    common.sanity_checks_for_pypi_installation()
    common.load_framework_extension("torch_nv")
    assert loader.loads == 0


def test_monolithic_core_is_not_an_empty_metapackage(common, loader):
    common.sanity_checks_for_pypi_installation()


def test_empty_metapackage_still_fails(common, monkeypatch):
    def missing(framework):
        raise FileNotFoundError("no core")
    monkeypatch.setattr(common, "_get_shared_object_file", missing)
    with pytest.raises(RuntimeError, match="empty"):
        common.sanity_checks_for_pypi_installation()


def test_split_install_version_check_is_preserved(common, loader, monkeypatch):
    monkeypatch.setattr(common, "get_te_core_package_info", lambda: (True, "transformer-engine-cu13", "2.17.0"))
    monkeypatch.setattr(common, "version", lambda name: "2.20.2")
    with pytest.raises(AssertionError, match="version mismatch"):
        common.sanity_checks_for_pypi_installation()


def test_preimported_native_is_validated_without_replacing_public(common, loader):
    native = ModuleType("transformer_engine_torch_nv")
    native.get_cudnn_version = lambda: 91100
    sys.modules["transformer_engine_torch_nv"] = native
    with pytest.raises(RuntimeError, match="cuDNN"):
        common.load_framework_extension("torch_nv")
    assert sys.modules["transformer_engine_torch"] is loader.stub
    native.get_cudnn_version = lambda: 91300
    common.load_framework_extension("torch_nv")
    assert loader.loads == 0
    assert sys.modules["transformer_engine_torch_nv"] is native


def test_loader_uses_real_nv_module_name(common, loader):
    common.load_framework_extension("torch_nv")
    assert loader.native.__name__ == "transformer_engine_torch_nv"
    assert sys.modules["transformer_engine_torch"] is loader.stub


def test_cuda_discovery_respects_runtime_disable(common, monkeypatch):
    # Exercise the discovery predicate without importing a real CUDA backend.
    path = Path(__file__).resolve().parents[1] / "transformer_engine/plugin/core/backends/vendor/cuda/cuda.py"
    tree = ast.parse(path.read_text())
    predicate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_check_cuda_available")
    code = compile(ast.Module(body=[predicate], type_ignores=[]), str(path), "exec")
    monkeypatch.setitem(sys.modules, "transformer_engine.common", common)
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
    def forbidden():
        pytest.fail("disabled CUDA backend attempted native library loading")
    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)), "_ensure_cuda_libs": forbidden}
    exec(code, namespace)
    assert namespace["_check_cuda_available"]() is False
