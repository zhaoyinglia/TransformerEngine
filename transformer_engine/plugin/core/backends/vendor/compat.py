"""Compatibility helpers for vendor Transformer Engine extensions."""

import warnings


_WARNED_OPERATIONS = set()


def call_v220_optional(extension, vendor, operation, *args, **kwargs):
    """Forward new v2.20.2 options or report an older vendor ABI explicitly."""
    try:
        return getattr(extension, operation)(*args, **kwargs)
    except AttributeError as error:
        raise NotImplementedError(
            f"{vendor} extension does not provide {operation} from the v2.20.2 API"
        ) from error
    except TypeError as error:
        if not _is_argument_mismatch(error):
            raise
        raise NotImplementedError(
            f"{vendor} extension does not support the v2.20.2 parameters for {operation}"
        ) from error


def _is_argument_mismatch(error):
    message = str(error)
    return any(
        marker in message
        for marker in (
            "incompatible function arguments",
            "positional arguments but",
            "required positional argument",
            "unexpected keyword argument",
            "takes exactly",
            "expected at most",
        )
    )


def _warn_vendor_upgrade(vendor, operation):
    key = (vendor, operation)
    if key in _WARNED_OPERATIONS:
        return
    _WARNED_OPERATIONS.add(key)
    warnings.warn(
        f"{vendor} Transformer Engine does not yet provide the upstream v2.17 {operation} ABI; the"
        " vendor TE is being upgraded; FL will use a semantics-preserving compatibility path when"
        " available.",
        RuntimeWarning,
        stacklevel=4,
    )


def _legacy_arguments(vendor, operation, args):
    if operation == "generic_gemm":
        if vendor == "Enflame":
            return ((*args[:12], args[14], *args[18:]),)
        return ()
    if operation in {"group_quantize", "bgrad_group_quantize"}:
        if args[-1] is not None:
            return ()
        if vendor in {"Hygon", "MetaX", "Iluvatar"}:
            return (args[:2],)
        if vendor in {"MUSA", "Enflame"}:
            return (args[:-1],)
        return ()
    if operation == "clamped_swiglu":
        return (args[:-1],) if args[-1] == 1.0 else ()
    if operation == "clamped_dswiglu":
        return (args[:-1],) if args[-1] == 1.0 else ()
    if operation in {
        "fused_topk_with_score_function_fwd",
        "fused_score_for_moe_aux_loss_fwd",
    }:
        return (args[:-1],) if args[-1] == 0 else ()
    if operation == "fused_topk_with_score_function_bwd" and args[-1] == 0:
        grad_logits = args[3]
        return ((grad_logits.shape[0], grad_logits.shape[1], *args[:-1]),)
    if operation == "fused_score_for_moe_aux_loss_bwd":
        grad_logits = args[2]
        return ((grad_logits.shape[0], grad_logits.shape[1], *args),)
    return ()


def call_legacy_fused_attn_backend(extension, vendor, params):
    """Use the old selector only when the new config has legacy semantics.

    The selector returns (backend, rejection reason). A None backend means the
    vendor's old ABI cannot decide this configuration without dropping fields.
    """
    def unsupported(field):
        return None, f"{vendor} fused attention selector cannot represent {field}"

    qkv_dtype = int(params.qkv_dtype)
    for field in ("o_dtype", "do_dtype", "dqkv_dtype"):
        if int(getattr(params, field)) != qkv_dtype:
            return unsupported(field)
    if int(params.scaling_mode) != 0:
        return unsupported("scaling_mode")

    layout_name = params.qkv_layout.name
    if "Paged_KV" in layout_name:
        return unsupported("paged KV layout")
    if layout_name.startswith(("NVTE_SB", "NVTE_BS", "NVTE_T", "NVTE_BH")):
        expected_format = (
            "NVTE_SBHD" if layout_name.startswith("NVTE_SB") else
            "NVTE_BSHD" if layout_name.startswith("NVTE_BS") else
            "NVTE_THD" if layout_name.startswith("NVTE_T") else "NVTE_BHSD"
        )
    else:
        return unsupported("qkv_layout")
    for field in ("o_format", "do_format"):
        format_name = getattr(params, field).name
        if format_name not in (expected_format, "NVTE_QKV_Format_NOT_SET"):
            return unsupported(field)
    if params.dqkv_layout.name not in (layout_name, "NVTE_QKV_Layout_NOT_SET"):
        return unsupported("dqkv_layout")
    for field in ("qkv_scale_inv_format", "do_scale_inv_format"):
        if getattr(params, field).name != "NVTE_QKV_Format_NOT_SET":
            return unsupported(field)
    for field in (
        "num_pages_k", "num_pages_v", "page_size_k", "page_size_v",
        "max_pages_per_seq_k", "max_pages_per_seq_v",
    ):
        if getattr(params, field):
            return unsupported(field)
    if expected_format == "NVTE_THD" and (params.num_tokens_q or params.num_tokens_kv):
        return unsupported("ragged token counts")
    for field in ("bias_batch_size", "bias_num_heads", "bias_seqlen_q", "bias_seqlen_kv"):
        if getattr(params, field):
            return unsupported(field)
    if not params.bottom_right_diagonal and int(params.attn_mask_type) in (4, 5):
        return unsupported("bottom_right_diagonal")

    def enum(name, value):
        return getattr(extension, name)(int(value))

    result = extension.get_fused_attn_backend(
        params.is_training,
        enum("DType", params.qkv_dtype),
        enum("DType", params.qkv_dtype),
        enum("NVTE_QKV_Layout", params.qkv_layout),
        enum("NVTE_Bias_Type", params.bias_type),
        enum("NVTE_Mask_Type", params.attn_mask_type),
        enum("NVTE_Softmax_Type", params.softmax_type),
        params.dropout,
        params.num_attn_heads,
        params.num_gqa_groups,
        params.max_seqlen_q,
        params.max_seqlen_kv,
        params.head_dim_qk,
        params.head_dim_v,
        params.window_size_left,
        params.window_size_right,
        params.return_max_logit,
        params.cuda_graph,
        params.deterministic,
    )
    if isinstance(result, tuple):
        return int(result[0]), result[1]
    return int(result), ""


class VendorTECompat:
    """Prefer the v2.17 ABI and fall back to the last FL ABI when equivalent."""

    _UPGRADED_OPERATIONS = {
        "generic_gemm",
        "group_quantize",
        "bgrad_group_quantize",
        "clamped_swiglu",
        "clamped_dswiglu",
        "fused_attn_fwd",
        "fused_attn_bwd",
        "fused_topk_with_score_function_fwd",
        "fused_topk_with_score_function_bwd",
        "fused_score_for_moe_aux_loss_fwd",
        "fused_score_for_moe_aux_loss_bwd",
    }

    def __init__(self, extension, vendor):
        self._extension = extension
        self._vendor = vendor

    def __getattr__(self, name):
        attribute = getattr(self._extension, name)
        if name not in self._UPGRADED_OPERATIONS or not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            # These wrappers supply the v2.17 positional contract. v2.20.2
            # inserted parameters before offsets / routing_map_format, so use
            # names first to avoid a successful call with shifted meanings.
            named_args = None
            if not kwargs and name in {"group_quantize", "bgrad_group_quantize"} and len(args) == 5:
                named_args = (args[:4], {"tensor_offsets": args[4]})
            elif not kwargs and name == "fused_topk_with_score_function_bwd" and len(args) == 9:
                named_args = (args[:8], {"routing_map_format": args[8]})
            if named_args is not None:
                try:
                    return attribute(*named_args[0], **named_args[1])
                except TypeError as error:
                    if not _is_argument_mismatch(error):
                        raise
            try:
                return attribute(*args, **kwargs)
            except TypeError as error:
                if kwargs or not _is_argument_mismatch(error):
                    raise
                legacy_candidates = _legacy_arguments(self._vendor, name, args)
                _warn_vendor_upgrade(self._vendor, name)
                if not legacy_candidates:
                    raise
                for legacy_args in legacy_candidates:
                    try:
                        return attribute(*legacy_args)
                    except TypeError as legacy_error:
                        if not _is_argument_mismatch(legacy_error):
                            raise
                raise

        return call


def vendor_te_compat(extension, vendor):
    """Wrap a vendor extension with the f031cf87-to-v2.17 ABI bridge."""
    return VendorTECompat(extension, vendor)
