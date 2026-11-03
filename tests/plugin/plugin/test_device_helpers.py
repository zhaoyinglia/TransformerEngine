"""Device selection remains dynamic and traceable for compiled training."""

import torch
import transformer_engine as te


def test_device_helpers_track_vendor_patch_in_fullgraph(monkeypatch):
    def forward(value):
        device_offset = 1 if te.te_device_type() == "cuda" else 2
        platform_offset = 3 if te.te_platform() is torch.cuda else 4
        return value + device_offset + platform_offset

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    value = torch.ones(2)
    monkeypatch.setattr(te, "TE_DEVICE_TYPE", "cuda")
    monkeypatch.setattr(te, "TE_PLATFORM", torch.cuda)
    torch.testing.assert_close(compiled(value), value + 4)
    monkeypatch.setattr(te, "TE_DEVICE_TYPE", "npu")
    monkeypatch.setattr(te, "TE_PLATFORM", torch.cpu)
    torch.testing.assert_close(compiled(value), value + 6)


def test_device_helpers_preserve_defaults_without_vendor_globals(monkeypatch):
    monkeypatch.delattr(te, "TE_DEVICE_TYPE", raising=False)
    monkeypatch.delattr(te, "TE_PLATFORM", raising=False)
    assert te.te_device_type() == "cuda"
    assert te.te_device_type("cpu") == "cpu"
    assert te.te_platform() is torch.cuda
    fallback = object()
    assert te.te_platform(fallback) is fallback
    monkeypatch.delattr(te, "torch", raising=False)
    assert te.te_platform() is None
    assert te.te_platform(fallback) is fallback
