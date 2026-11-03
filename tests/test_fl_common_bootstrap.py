"""CPU regressions for FL common initialization, without loading CUDA libraries."""

import ast
import os
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
    # Isolate FL registration from real hardware discovery.
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
def bootstrap():
    path = Path(__file__).resolve().parents[1] / "transformer_engine/plugin/bootstrap.py"
    spec = importlib.util.spec_from_file_location("_fl_bootstrap_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    core.load_plugins = lambda: sys.modules.update(transformer_engine_torch=core.get_tefl_module())
    monkeypatch.setenv("NVTE_PLUGIN", core.__name__)
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
    state.stub = stub
    state.core = core
    return state


def test_native_and_fl_aliases_are_stable(common, loader):
    common.load_framework_extension("torch")
    assert sys.modules["transformer_engine_torch"] is loader.stub
    assert sys.modules["transformer_engine_torch_nv"] is loader.native
    common.load_framework_extension("torch")
    assert loader.loads == 1


@pytest.mark.parametrize("failure", ["exec", "cudnn"])
def test_initialization_errors_propagate(common, loader, failure):
    if failure == "exec":
        loader.error = ImportError("broken native load")
    elif failure == "cudnn":
        loader.version = 91100
    with pytest.raises((ImportError, RuntimeError)):
        common.load_framework_extension("torch")


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
            common.load_framework_extension("torch")
    else:
        common.load_framework_extension("torch")
    if failure:
        assert "transformer_engine_torch_nv" not in sys.modules
    else:
        assert sys.modules["transformer_engine_torch_nv"] is loader.native
    assert sys.modules["transformer_engine_torch"] is (loader.native if failure else replacement)


def test_empty_override_uses_build_metadata(bootstrap, monkeypatch, tmp_path):
    root = tmp_path / "transformer_engine"
    (root / "common").mkdir(parents=True)
    (root / "plugin/core").mkdir(parents=True)
    (root / "plugin/core/_build_config.py").write_text("SKIP_CUDA_BUILD = True\n")
    monkeypatch.setattr(bootstrap, "__file__", str(root / "plugin/bootstrap.py"))
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "")
    assert bootstrap.skip_cuda_build() is True
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "0")
    assert bootstrap.skip_cuda_build() is False


def test_no_cuda_skips_native_loader(common, bootstrap, loader, monkeypatch):
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
    bootstrap.load_pytorch_extension()
    assert loader.loads == 0
    assert sys.modules["transformer_engine_torch"] is loader.stub


@pytest.mark.parametrize("release", [None, "1"])
@pytest.mark.parametrize("failure", [False, True])
def test_common_import_guard_is_temporary(bootstrap, monkeypatch, release, failure):
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
    monkeypatch.delenv("NVTE_PROJECT_BUILDING", raising=False)
    if release is None:
        monkeypatch.delenv("NVTE_RELEASE_BUILD", raising=False)
    else:
        monkeypatch.setenv("NVTE_RELEASE_BUILD", release)
    def import_common(name):
        assert name == "transformer_engine.common"
        assert os.environ["NVTE_PROJECT_BUILDING"] == "1"
        assert os.environ["NVTE_RELEASE_BUILD"] == "0"
        if failure:
            raise RuntimeError("import failed")
    monkeypatch.setattr(bootstrap.importlib, "import_module", import_common)
    if failure:
        with pytest.raises(RuntimeError, match="import failed"):
            bootstrap.import_common()
    else:
        bootstrap.import_common()
    assert "NVTE_PROJECT_BUILDING" not in os.environ
    assert os.environ.get("NVTE_RELEASE_BUILD") == release


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


def test_loader_preserves_native_under_nv_alias(common, loader):
    common.load_framework_extension("torch")
    assert loader.native.__name__ == "transformer_engine_torch"
    assert sys.modules["transformer_engine_torch_nv"] is loader.native
    assert sys.modules["transformer_engine_torch"] is loader.stub


def test_cuda_discovery_respects_runtime_disable(bootstrap, monkeypatch):
    # Exercise the discovery predicate without importing a real CUDA backend.
    path = Path(__file__).resolve().parents[1] / "transformer_engine/plugin/core/backends/vendor/cuda/cuda.py"
    tree = ast.parse(path.read_text())
    predicate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_check_cuda_available")
    code = compile(ast.Module(body=[predicate], type_ignores=[]), str(path), "exec")
    monkeypatch.setitem(sys.modules, "transformer_engine.plugin.bootstrap", bootstrap)
    monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
    def forbidden():
        pytest.fail("disabled CUDA backend attempted native library loading")
    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)), "_ensure_cuda_libs": forbidden}
    exec(code, namespace)
    assert namespace["_check_cuda_available"]() is False


def test_native_alias_exists_before_fl_setup(common, loader):
    def setup():
        assert sys.modules["transformer_engine_torch_nv"] is loader.native
        assert sys.modules["transformer_engine_torch"] is loader.native
        return loader.stub
    loader.core.get_tefl_module = setup
    common.load_framework_extension("torch")
    assert sys.modules["transformer_engine_torch"] is loader.stub


def test_plugin_bootstrap_defers_core_and_preserves_exports(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "transformer_engine"
    parent = ModuleType("transformer_engine")
    parent.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "transformer_engine", parent)
    for name in list(sys.modules):
        if name == "transformer_engine.plugin" or name.startswith("transformer_engine.plugin."):
            monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("transformer_engine.plugin", root / "plugin/__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    spec = importlib.util.spec_from_file_location("transformer_engine.plugin.bootstrap", root / "plugin/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, bootstrap)
    spec.loader.exec_module(bootstrap)
    assert "transformer_engine.plugin.core" not in sys.modules

    core = ModuleType("transformer_engine.plugin.core")
    stub = ModuleType("fl_dispatch")
    core.TEFLBackendBase = type("TEFLBackendBase", (), {})
    core.TEFLModule = type("TEFLModule", (), {})
    core.get_tefl_module = lambda: stub
    core.get_registry = lambda: "registry"
    monkeypatch.setitem(sys.modules, core.__name__, core)
    for name in plugin.__all__:
        assert getattr(plugin, name) is (stub if name == "tefl" else getattr(core, name))
    with pytest.raises(AttributeError):
        getattr(plugin, "unknown_export")
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", ModuleType("native"))
    plugin.load_plugins()
    assert sys.modules["transformer_engine_torch"] is stub


@pytest.mark.parametrize("no_cuda", [False, True])
@pytest.mark.parametrize("has_patches", [False, True])
def test_optional_patches_run_after_plugin(common, bootstrap, loader, monkeypatch, no_cuda, has_patches):
    events = []
    loader.core.__path__ = []
    def setup():
        events.append("load")
        return loader.stub
    loader.core.get_tefl_module = setup
    patch = ModuleType(loader.core.__name__ + ".patches")
    patch.__spec__ = ModuleSpec(patch.__name__, loader=None)
    def apply():
        assert sys.modules["transformer_engine_torch"] is loader.stub
        events.append("patch")
    patch.apply_patches = apply
    if has_patches:
        monkeypatch.setitem(sys.modules, patch.__name__, patch)
    if no_cuda:
        monkeypatch.setenv("TE_FL_SKIP_CUDA", "1")
        bootstrap.load_pytorch_extension()
    else:
        common.load_framework_extension("torch")
    assert events == (["load", "patch"] if has_patches else ["load"])


@pytest.mark.parametrize("patched", [False, True])
def test_device_defaults_preserve_plugin_values(monkeypatch, patched):
    parent = ModuleType("transformer_engine")
    platform = object()
    if patched:
        parent.TE_DEVICE_TYPE = "test_vendor"
        parent.TE_PLATFORM = platform
    path = Path(__file__).resolve().parents[1] / "transformer_engine/pytorch/__init__.py"
    source = path.read_text()
    start = source.index("# Device type and platform globals")
    end = source.index("from transformer_engine.pytorch import constants", start)
    cuda = object()
    namespace = {"transformer_engine": parent, "torch": SimpleNamespace(cuda=cuda)}
    exec(compile(source[start:end], str(path), "exec"), namespace)
    assert parent.te_device_type() == ("test_vendor" if patched else "cuda")
    assert parent.te_platform() is (platform if patched else cuda)
