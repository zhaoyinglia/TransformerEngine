# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

set -x

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

FL_TEST_SUITE=L0
source "$TE_PATH/qa/fl_test_policy.sh"

# L0 keeps one mature FlashAttention generation; L3 owns newer-generation coverage.
export NVTE_FLASH_ATTN_V2=1
export NVTE_FLASH_ATTN_V3=0
export NVTE_FLASH_ATTN_V4=0

pip3 install pytest==8.2.1 expecttest || error_exit "Failed to install pytest"

run_test_step "$TE_PATH/tests/pytorch/test_sanity.py" "test_sanity.py" env NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py || test_fail "test_sanity.py"
run_test_step "$TE_PATH/tests/pytorch/test_recipe.py" "test_recipe.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py || test_fail "test_recipe.py"
run_test_step "$TE_PATH/tests/pytorch/test_custom_recipe.py" "test_custom_recipe.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_custom_recipe.xml $TE_PATH/tests/pytorch/test_custom_recipe.py || test_fail "test_custom_recipe.py"
run_test_step "$TE_PATH/tests/pytorch/test_deferred_init.py" "test_deferred_init.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py || test_fail "test_deferred_init.py"
run_test_step "$TE_PATH/tests/pytorch/test_numerics.py" "test_numerics.py" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py || test_fail "test_numerics.py"
run_test_step "$TE_PATH/tests/pytorch/test_cuda_graphs.py" "test_cuda_graphs.py" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py || test_fail "test_cuda_graphs.py"
run_test_step "$TE_PATH/tests/pytorch/test_jit.py" "test_jit.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py || test_fail "test_jit.py"
run_test_step "$TE_PATH/tests/pytorch/test_fused_rope.py" "test_fused_rope.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py || test_fail "test_fused_rope.py"
run_test_step "$TE_PATH/tests/pytorch/nvfp4" "test_nvfp4" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4.xml $TE_PATH/tests/pytorch/nvfp4 || test_fail "test_nvfp4"
run_test_step "$TE_PATH/tests/pytorch/mxfp8" "test_mxfp8" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8.xml $TE_PATH/tests/pytorch/mxfp8 || test_fail "test_mxfp8"
run_test_step "$TE_PATH/tests/pytorch/test_weight_swizzle_in_layers.py" "test_weight_swizzle_in_layers.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_weight_swizzle_in_layers.xml $TE_PATH/tests/pytorch/test_weight_swizzle_in_layers.py || test_fail "test_weight_swizzle_in_layers.py"
run_test_step "$TE_PATH/tests/pytorch/test_quantized_tensor.py" "test_quantized_tensor.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_quantized_tensor.xml $TE_PATH/tests/pytorch/test_quantized_tensor.py || test_fail "test_quantized_tensor.py"
run_test_step "$TE_PATH/tests/pytorch/test_torch_compile.py" "test_torch_compile.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_torch_compile.xml $TE_PATH/tests/pytorch/test_torch_compile.py || test_fail "test_torch_compile.py"
run_test_step "$TE_PATH/tests/pytorch/test_float8blockwisetensor.py" "test_float8blockwisetensor.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py || test_fail "test_float8blockwisetensor.py"
run_test_step "$TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" "test_float8_blockwise_scaling_exact.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py || test_fail "test_float8_blockwise_scaling_exact.py"
run_test_step "$TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" "test_float8_blockwise_gemm_exact.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py || test_fail "test_float8_blockwise_gemm_exact.py"
run_test_step "$TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py" "test_float8_current_scaling_exact.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_current_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py || test_fail "test_float8_current_scaling_exact.py"
run_test_step "$TE_PATH/tests/pytorch/test_grouped_tensor.py" "test_grouped_tensor.py" env NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_grouped_tensor.xml $TE_PATH/tests/pytorch/test_grouped_tensor.py || test_fail "test_grouped_tensor.py"
run_test_step "$TE_PATH/tests/pytorch/test_gqa.py" "test_gqa.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py || test_fail "test_gqa.py"
run_test_step "$TE_PATH/tests/pytorch/test_qk_norm.py" "test_qk_norm.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_qk_norm.xml $TE_PATH/tests/pytorch/test_qk_norm.py || test_fail "test_qk_norm.py"
run_test_step "$TE_PATH/tests/pytorch/test_fused_optimizer.py" "test_fused_optimizer.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py || test_fail "test_fused_optimizer.py"
run_test_step "$TE_PATH/tests/pytorch/test_multi_tensor.py" "test_multi_tensor.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_multi_tensor.xml $TE_PATH/tests/pytorch/test_multi_tensor.py || test_fail "test_multi_tensor.py"
run_test_step "$TE_PATH/tests/pytorch/test_fusible_ops.py" "test_fusible_ops.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/test_fusible_ops.py || test_fail "test_fusible_ops.py"
run_test_step "$TE_PATH/tests/pytorch/layernorm_mlp/test_selective_activation_checkpoint.py" "test_selective_activation_checkpoint.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_selective_activation_checkpoint.xml $TE_PATH/tests/pytorch/layernorm_mlp/test_selective_activation_checkpoint.py || test_fail "test_selective_activation_checkpoint.py"
run_test_step "$TE_PATH/tests/pytorch/test_distributed_weight.py" "test_distributed_weight.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_distributed_weight.xml $TE_PATH/tests/pytorch/test_distributed_weight.py || test_fail "test_distributed_weight.py"
run_test_step "$TE_PATH/tests/pytorch/test_backward_override.py" "test_backward_override.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backward_override.xml $TE_PATH/tests/pytorch/test_backward_override.py || test_fail "test_backward_override.py"
run_test_step "$TE_PATH/tests/pytorch/test_permutation.py" "test_permutation.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_permutation.xml $TE_PATH/tests/pytorch/test_permutation.py || test_fail "test_permutation.py"
run_test_step "$TE_PATH/tests/pytorch/test_cross_entropy.py" "test_cross_entropy.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cross_entropy.xml $TE_PATH/tests/pytorch/test_cross_entropy.py || test_fail "test_cross_entropy.py"
run_test_step "$TE_PATH/tests/pytorch/test_cpu_offloading.py" "test_cpu_offloading.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py || test_fail "test_cpu_offloading.py"
run_test_step "$TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" "test_cpu_offloading_v1.py" env NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading_v1.xml $TE_PATH/tests/pytorch/test_cpu_offloading_v1.py || test_fail "test_cpu_offloading_v1.py"
run_test_step "$TE_PATH/tests/pytorch/test_hybrid_quantization.py" "test_hybrid_quantization.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hybrid_quantization.xml $TE_PATH/tests/pytorch/test_hybrid_quantization.py || test_fail "test_hybrid_quantization.py"
run_test_step "$TE_PATH/tests/pytorch/test_identity_quantizer.py" "test_identity_quantizer.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_identity_quantizer.xml $TE_PATH/tests/pytorch/test_identity_quantizer.py || test_fail "test_identity_quantizer.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_attention.py" "test_attention.py" env NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention.xml $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_flex_attention.py" "test_flex_attention.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_flex_attention.xml $TE_PATH/tests/pytorch/attention/test_flex_attention.py || test_fail "test_flex_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_gdn_attention.py" "test_gdn_attention.py" env NVTE_GDN_TEST_REQUIRED=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdn_attention.xml $TE_PATH/tests/pytorch/attention/test_gdn_attention.py || test_fail "test_gdn_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_gdn2_attention.py" "test_gdn2_attention.py" env NVTE_GDN2_TEST_REQUIRED=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdn2_attention.xml $TE_PATH/tests/pytorch/attention/test_gdn2_attention.py || test_fail "test_gdn2_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_gdp_attention.py" "test_gdp_attention.py" env NVTE_GDP_TEST_REQUIRED=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdp_attention.xml $TE_PATH/tests/pytorch/attention/test_gdp_attention.py || test_fail "test_gdp_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_attention.py" "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_attention.py" env NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention_deterministic.xml $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py" "test_linear_mxfp8_attention.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_linear_mxfp8_attention.xml $TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py || test_fail "test_linear_mxfp8_attention.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_fused_mla_q_uproj.py" "test_fused_mla_q_uproj.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_mla_q_uproj.xml $TE_PATH/tests/pytorch/attention/test_fused_mla_q_uproj.py || test_fail "test_fused_mla_q_uproj.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_kv_cache.py" "test_kv_cache.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_kv_cache.xml $TE_PATH/tests/pytorch/attention/test_kv_cache.py || test_fail "test_kv_cache.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py" "test_cu_seqlens_cache.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cu_seqlens_cache.xml $TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py || test_fail "test_cu_seqlens_cache.py"
run_test_step "$TE_PATH/tests/pytorch/attention/test_mixed_thd_attention.py" "test_mixed_thd_attention.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mixed_thd_attention.xml $TE_PATH/tests/pytorch/attention/test_mixed_thd_attention.py || test_fail "test_mixed_thd_attention.py"
run_test_step "$TE_PATH/tests/pytorch/test_hf_integration.py" "test_hf_integration.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hf_integration.xml $TE_PATH/tests/pytorch/test_hf_integration.py || test_fail "test_hf_integration.py"
export NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint
if ! fl_skip_reason "$TE_PATH/tests/pytorch/test_checkpoint.py" && [ ! -d "$NVTE_TEST_CHECKPOINT_ARTIFACT_PATH" ]; then
    python3 $TE_PATH/tests/pytorch/test_checkpoint.py --save-checkpoint all || error_exit "Failed to generate checkpoint files"
fi
run_test_step "$TE_PATH/tests/pytorch/test_checkpoint.py" "test_checkpoint.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_checkpoint.xml $TE_PATH/tests/pytorch/test_checkpoint.py || test_fail "test_checkpoint.py"
run_test_step "$TE_PATH/tests/pytorch/test_fused_router.py" "test_fused_router.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_router.xml $TE_PATH/tests/pytorch/test_fused_router.py || test_fail "test_fused_router.py"
run_test_step "$TE_PATH/tests/pytorch/test_partial_cast.py" "test_partial_cast.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_partial_cast.xml $TE_PATH/tests/pytorch/test_partial_cast.py || test_fail "test_partial_cast.py"
run_test_step "$TE_PATH/tests/pytorch/test_mxfp8_2d_quantize.py" "test_mxfp8_2d_quantize.py" env python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8_2d_quantize.xml $TE_PATH/tests/pytorch/test_mxfp8_2d_quantize.py || test_fail "test_mxfp8_2d_quantize.py"
# Disable autotuning to make unittests faster. In addition, disable TF32 path to fully align with the pytorch reference implementation's precision
run_test_step "$TE_PATH/tests/pytorch/test_mhc.py" "test_mhc.py" env NVTE_DISABLE_TRITON_AUTOTUNING=1 NVIDIA_TF32_OVERRIDE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mhc.xml $TE_PATH/tests/pytorch/test_mhc.py || test_fail "test_mhc.py"
run_test_step "$TE_PATH/tests/pytorch/test_mhc.py" "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_mhc.py" env NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_DISABLE_TRITON_AUTOTUNING=1 NVIDIA_TF32_OVERRIDE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mhc_deterministic.xml $TE_PATH/tests/pytorch/test_mhc.py || test_fail "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_mhc.py"
run_test_step "$TE_PATH/tests/pytorch/test_grouped_linear.py" "test_grouped_linear.py" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_linear.xml $TE_PATH/tests/pytorch/test_grouped_linear.py || test_fail "test_grouped_linear.py"
run_test_step "$TE_PATH/tests/pytorch/test_ops_grouped_linear_distributed_weight.py" "test_ops_grouped_linear_distributed_weight.py" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_ops_grouped_linear_distributed_weight.xml $TE_PATH/tests/pytorch/test_ops_grouped_linear_distributed_weight.py || test_fail "test_ops_grouped_linear_distributed_weight.py"
run_test_step "$TE_PATH/tests/pytorch/test_grouped_mlp.py" "test_grouped_mlp.py" env NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_mlp.xml $TE_PATH/tests/pytorch/test_grouped_mlp.py || test_fail "test_grouped_mlp.py"


# Fork plugin coverage plus v2.20.2 ABI regression tests.
PLUGIN_TEST_ROOT="$TE_PATH/tests/plugin"
run_test_step "$PLUGIN_TEST_ROOT/plugin/test_policy.py" "plugin pytest_test_plugin_policy.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_policy.xml $PLUGIN_TEST_ROOT/plugin/test_policy.py || test_fail "plugin pytest_test_plugin_policy.xml"
run_test_step "$PLUGIN_TEST_ROOT/plugin/test_manager.py" "plugin pytest_test_plugin_manager.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_manager.xml $PLUGIN_TEST_ROOT/plugin/test_manager.py || test_fail "plugin pytest_test_plugin_manager.xml"
run_test_step "$PLUGIN_TEST_ROOT/plugin/test_policy_selection.py" "plugin pytest_test_plugin_policy_selection.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_policy_selection.xml $PLUGIN_TEST_ROOT/plugin/test_policy_selection.py || test_fail "plugin pytest_test_plugin_policy_selection.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_lifecycle.py" "plugin pytest_test_backend_flagos.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos.xml $PLUGIN_TEST_ROOT/backend/flagos/test_lifecycle.py || test_fail "plugin pytest_test_backend_flagos.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_fused_rope.py" "plugin pytest_test_backend_flagos_fused_rope.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_fused_rope.xml $PLUGIN_TEST_ROOT/backend/flagos/test_fused_rope.py || test_fail "plugin pytest_test_backend_flagos_fused_rope.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_optimizer.py" "plugin pytest_test_backend_flagos_fused_adam.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_fused_adam.xml $PLUGIN_TEST_ROOT/backend/flagos/test_optimizer.py || test_fail "plugin pytest_test_backend_flagos_fused_adam.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_gemm.py" "plugin pytest_test_backend_flagos_gemm.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_gemm.xml $PLUGIN_TEST_ROOT/backend/flagos/test_gemm.py || test_fail "plugin pytest_test_backend_flagos_gemm.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_multi_tensor.py" "plugin pytest_test_backend_flagos_multi_tensor.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_multi_tensor.xml $PLUGIN_TEST_ROOT/backend/flagos/test_multi_tensor.py || test_fail "plugin pytest_test_backend_flagos_multi_tensor.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_rmsnorm.py" "plugin pytest_test_backend_flagos_rmsnorm.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_rmsnorm.xml $PLUGIN_TEST_ROOT/backend/flagos/test_rmsnorm.py || test_fail "plugin pytest_test_backend_flagos_rmsnorm.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/flagos/test_softmax.py" "plugin pytest_test_backend_flagos_softmax.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_softmax.xml $PLUGIN_TEST_ROOT/backend/flagos/test_softmax.py || test_fail "plugin pytest_test_backend_flagos_softmax.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/reference/test_lifecycle.py" "plugin pytest_test_backend_reference.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference.xml $PLUGIN_TEST_ROOT/backend/reference/test_lifecycle.py || test_fail "plugin pytest_test_backend_reference.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/reference/test_activation.py" "plugin pytest_test_backend_reference_activation.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_activation.xml $PLUGIN_TEST_ROOT/backend/reference/test_activation.py || test_fail "plugin pytest_test_backend_reference_activation.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/reference/test_dropout.py" "plugin pytest_test_backend_reference_dropout.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_dropout.xml $PLUGIN_TEST_ROOT/backend/reference/test_dropout.py || test_fail "plugin pytest_test_backend_reference_dropout.xml"
run_test_step "$PLUGIN_TEST_ROOT/backend/reference/test_gemm.py" "plugin pytest_test_backend_reference_gemm.xml" env python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_gemm.xml $PLUGIN_TEST_ROOT/backend/reference/test_gemm.py || test_fail "plugin pytest_test_backend_reference_gemm.xml"
run_test_step "$PLUGIN_TEST_ROOT/plugin/test_v220_api_contract.py" "v2.20.2 API contract" python3 -m pytest --junitxml="$XML_LOG_DIR/pytest_v220_api.xml" "$PLUGIN_TEST_ROOT/plugin/test_v220_api_contract.py" "$PLUGIN_TEST_ROOT/plugin/test_vendor_compat.py" "$PLUGIN_TEST_ROOT/plugin/test_device_helpers.py" || test_fail "v2.20.2 API contract"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "Executed test commands passed. Exclusions, blocked groups and filters: $FL_TEST_RESULTS"
exit 0
