"""CPU regressions for v2.20.2 argument binding and optional-op dispatch."""

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from transformer_engine.plugin.core.backends.vendor.compat import VendorTECompat, call_v220_optional
from transformer_engine.plugin.core.ops import TEFLModule, TEFLBackendBase


@pytest.mark.parametrize("operation", ["group_quantize", "bgrad_group_quantize"])
def test_vendor_offsets_keep_their_meaning_on_new_native_abi(operation):
    def native(tensor, quantizer, num_tensors, first_dims, last_dims=None, tensor_offsets=None):
        return last_dims, tensor_offsets

    offsets = object()
    compat = VendorTECompat(SimpleNamespace(**{operation: native}), "TestVendor")
    assert getattr(compat, operation)(None, None, 2, [3, 4], offsets) == (None, offsets)


def test_vendor_router_format_does_not_become_dense_indices():
    def native(*args, use_dense_indices=False, routing_map_format=0):
        assert len(args) == 8
        return use_dense_indices, routing_map_format

    compat = VendorTECompat(SimpleNamespace(fused_topk_with_score_function_bwd=native), "TestVendor")
    assert compat.fused_topk_with_score_function_bwd(*([None] * 8), 2) == (False, 2)


@pytest.mark.parametrize("vendor", ["cuda", "hygon", "metax", "iluvatar", "musa", "enflame"])
@pytest.mark.parametrize("operation", ["group_quantize", "bgrad_group_quantize"])
def test_group_public_signature_and_forwarding(vendor, operation):
    module = importlib.import_module(f"transformer_engine.plugin.core.backends.vendor.{vendor}.{vendor}")
    backend_class = next(value for value in vars(module).values()
                         if isinstance(value, type) and value is not TEFLBackendBase
                         and issubclass(value, TEFLBackendBase) and value.__module__ == module.__name__)
    backend = object.__new__(backend_class)
    calls = []
    native = SimpleNamespace(**{operation: lambda *args, **kwargs: calls.append((args, kwargs))})
    backend._get_tex = lambda: native
    function = getattr(backend, operation)
    assert list(inspect.signature(function).parameters)[:6] == [
        "tensor", "quantizer", "num_tensors", "first_dims", "last_dims", "tensor_offsets"]
    last_dims, offsets = object(), object()
    function(None, None, 2, [3, 4], last_dims, offsets)
    assert calls[0][1]["last_dims"] is last_dims
    assert calls[0][1]["tensor_offsets"] is offsets


def test_optional_arguments_are_not_discarded_on_old_extension():
    native = SimpleNamespace(group_quantize=lambda tensor, quantizer: None)
    with pytest.raises(NotImplementedError, match="v2.20.2 parameters"):
        call_v220_optional(native, "TestVendor", "group_quantize", None, None, last_dims=[2])


def test_native_alias_does_not_override_registered_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine_torch_nv",
                        SimpleNamespace(scaled_swiglu=lambda: "wrong native"))
    manager = SimpleNamespace(
        ensure_initialized=lambda: None,
        registry=SimpleNamespace(list_operators=lambda: ["scaled_swiglu"]),
        call=lambda name, *args, **kwargs: (name, "selected vendor"),
    )
    assert TEFLModule(manager).scaled_swiglu() == ("scaled_swiglu", "selected vendor")


def test_unregistered_native_alias_is_not_a_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine_torch_nv",
                        SimpleNamespace(scaled_swiglu=lambda: "wrong native"))
    manager = SimpleNamespace(ensure_initialized=lambda: None,
                              registry=SimpleNamespace(list_operators=lambda: []))
    module = TEFLModule(manager)
    with pytest.raises(NotImplementedError, match="no registered implementation"):
        module.scaled_swiglu()
    with pytest.raises(NotImplementedError, match="no registered implementation"):
        module.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights()


def test_cuda_registration_preserves_optional_native_operations(monkeypatch):
    from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend
    from transformer_engine.plugin.core.backends.vendor.cuda.register_ops import register_builtins
    from transformer_engine.plugin.core.registry import OpRegistry

    native = SimpleNamespace(
        scaled_swiglu=lambda *args: args,
        copy_data_ptrs_to_device=lambda *args: args,
        splits_to_offsets_multi=lambda *args: args,
        grouped_mlp_experimental=SimpleNamespace(
            swizzle_scales_and_pack_ptrs_for_discrete_weights=lambda *args: args),
    )
    monkeypatch.setattr(CUDABackend, "is_available", lambda self: True)
    monkeypatch.setattr(CUDABackend, "_get_tex", lambda self: native)
    registry = OpRegistry()
    register_builtins(registry)
    assert "scaled_swiglu" in registry.list_operators()
    assert "copy_data_ptrs_to_device" in registry.list_operators()
    assert "splits_to_offsets_multi" in registry.list_operators()
    assert "grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights" in registry.list_operators()


def test_nvfp4_group_preserves_pybind_positional_order():
    from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend

    calls = []
    native = SimpleNamespace(nvfp4_group_quantize_with_amax=lambda *args, **kwargs: calls.append((args, kwargs)))
    backend = CUDABackend()
    backend._get_tex = lambda: native
    values = tuple(object() for _ in range(8))
    backend.nvfp4_group_quantize_with_amax(*values)
    assert calls == [(values, {})]


@pytest.mark.parametrize("use_output_buffer", [False, True])
def test_cuda_swap_first_dims_uses_keyword_output_buffer(use_output_buffer):
    from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend

    tensor = object()
    output = object() if use_output_buffer else None
    result = object()

    def native(value, *, out):
        assert value is tensor
        assert out is output
        return result

    backend = CUDABackend()
    backend._get_tex = lambda: SimpleNamespace(swap_first_dims=native)
    assert backend.swap_first_dims(tensor, output) is result


@pytest.mark.parametrize("enabled", [False, True])
def test_cublasmp_build_query_is_exported_and_registered(monkeypatch, enabled):
    from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend
    from transformer_engine.plugin.core.backends.vendor.cuda.register_ops import register_builtins
    from transformer_engine.plugin.core.registry import OpRegistry

    native = SimpleNamespace(nvte_built_with_cublasmp=lambda: enabled)
    monkeypatch.setattr(CUDABackend, "is_available", lambda self: True)
    monkeypatch.setattr(CUDABackend, "_get_tex", lambda self: native)
    registry = OpRegistry()
    register_builtins(registry)
    implementation = registry.get_implementations("nvte_built_with_cublasmp")[0]
    manager = SimpleNamespace(
        ensure_initialized=lambda: None,
        registry=registry,
        call=lambda name: implementation.fn(),
    )
    module = TEFLModule(manager)
    assert "nvte_built_with_cublasmp" in module.__all__
    assert module.nvte_built_with_cublasmp() is enabled
