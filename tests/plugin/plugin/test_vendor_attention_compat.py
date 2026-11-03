"""Compatibility checks for pre-v2.20 vendor fused-attention selectors."""

from types import SimpleNamespace

import pytest

from transformer_engine.plugin.core import ops
from transformer_engine.plugin.core.backends.vendor.compat import call_legacy_fused_attn_backend


class LegacyExtension:
    DType = ops.DType
    NVTE_QKV_Layout = ops.NVTE_QKV_Layout
    NVTE_Bias_Type = ops.NVTE_Bias_Type
    NVTE_Mask_Type = ops.NVTE_Mask_Type
    NVTE_Softmax_Type = ops.NVTE_Softmax_Type

    def __init__(self):
        self.calls = []

    def get_fused_attn_backend(self, *args):
        self.calls.append(args)
        return ops.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen


def params(**overrides):
    values = dict(
        is_training=True, deterministic=False, cuda_graph=False, return_max_logit=False,
        attn_mask_type=ops.NVTE_Mask_Type.NVTE_NO_MASK,
        bias_type=ops.NVTE_Bias_Type.NVTE_NO_BIAS,
        window_size_left=-1, window_size_right=-1, bottom_right_diagonal=True,
        softmax_type=ops.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
        scaling_mode=ops.NVTEScalingMode.NVTE_DELAYED_TENSOR_SCALING,
        dropout=0.0, attn_scale=0.125,
        qkv_dtype=ops.DType.kBFloat16, o_dtype=ops.DType.kBFloat16,
        do_dtype=ops.DType.kBFloat16, dqkv_dtype=ops.DType.kBFloat16,
        qkv_layout=ops.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
        o_format=ops.NVTE_QKV_Format.NVTE_BSHD,
        do_format=ops.NVTE_QKV_Format.NVTE_BSHD,
        dqkv_layout=ops.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
        qkv_scale_inv_format=ops.NVTE_QKV_Format.NVTE_QKV_Format_NOT_SET,
        do_scale_inv_format=ops.NVTE_QKV_Format.NVTE_QKV_Format_NOT_SET,
        batch_size=2, num_attn_heads=8, num_gqa_groups=8,
        head_dim_qk=64, head_dim_v=64, max_seqlen_q=128, max_seqlen_kv=128,
        num_tokens_q=0, num_tokens_kv=0,
        num_pages_k=0, num_pages_v=0, page_size_k=0, page_size_v=0,
        max_pages_per_seq_k=0, max_pages_per_seq_v=0,
        bias_batch_size=0, bias_num_heads=0, bias_seqlen_q=0, bias_seqlen_kv=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_old_selector_receives_legacy_equivalent_configuration():
    extension = LegacyExtension()
    backend, reason = call_legacy_fused_attn_backend(extension, "test-vendor", params())
    assert backend == ops.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
    assert reason == ""
    assert len(extension.calls) == 1
    assert len(extension.calls[0]) == 19
    assert extension.calls[0][7] == 0.0


@pytest.mark.parametrize("changes,field", [
    ({"scaling_mode": ops.NVTEScalingMode.NVTE_BLOCK_SCALING_1D}, "scaling_mode"),
    ({"o_dtype": ops.DType.kFloat16}, "o_dtype"),
    ({"num_pages_k": 2}, "num_pages_k"),
    ({"bias_seqlen_q": 128}, "bias_seqlen_q"),
    ({"o_format": ops.NVTE_QKV_Format.NVTE_THD}, "o_format"),
])
def test_new_constraints_are_rejected_only_when_used(changes, field):
    extension = LegacyExtension()
    backend, reason = call_legacy_fused_attn_backend(extension, "test-vendor", params(**changes))
    assert backend is None
    assert field in reason
    assert not extension.calls
