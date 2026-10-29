#!/bin/bash


: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Upstream v2.20.2 L0 attention backend selection.
export NVTE_FLASH_ATTN_V2=1
export NVTE_FLASH_ATTN_V3=0
export NVTE_FLASH_ATTN_V4=0

pip install pytest==8.2.1 || exit 1
# solve test_fused_optimizer import error
pip install expecttest || exit 1
FAIL=0

IS_CUDA_BACKEND=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}


run_test_step() {
    local xml_file=$1
    local test_path=$2
    local cmd=$3
    local label=$4

    if [ "$PLATFORM" = "metax" ]; then
        case "$test_path" in
            *"test_numerics.py" | \
            *"test_sanity.py" | \
            *"test_cross_entropy.py" | \
            *"test_fused_rope.py" | \
            *"test_gqa.py" | \
            *"test_fused_optimizer.py" | \
            *"test_multi_tensor.py" | \
            *"test_cpu_offloading.py" | \
            *"test_cpu_offloading_v1.py" | \
            *"test_attention.py" | \
            *"attention/test_kv_cache.py" | \
            *"test_checkpoint.py" | \
            *"test_fused_router.py" | \
            *"test_cuda_graphs.py" | \
            *"test_hf_integration.py") # transformers library may not be available in CI
                echo "-------------------------------------------------------"
                echo "[SKIP] Platform MetaX: Ignoring $label"
                echo "-------------------------------------------------------"
                return 0
                ;;
        esac
    fi

    if [[ "$IS_CUDA_BACKEND" == *"cuda"* ]]; then
        # transformers library may not be available in CI
        if [[ "$test_path" == *"test_checkpoint.py" || "$test_path" == *"test_cpu_offloading.py" || "$test_path" == *"test_cpu_offloading_v1.py" || "$test_path" == *"test_attention.py" || "$test_path" == *"attention/test_kv_cache.py" || "$test_path" == *"test_hf_integration.py" ]]; then
            echo "-------------------------------------------------------"
            echo "[SKIP] CUDA Backend detected: Ignoring $label"
            echo "-------------------------------------------------------"
            return 0
        fi
    fi


    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"

    eval "$cmd" || test_fail "$label"
}


# Step: Sanity
if [ "$PLATFORM" = "metax" ]; then
    SANITY_CMD="NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py -k \"not (test_sanity_layernorm_mlp or test_sanity_gpt or test_sanity_bert or test_sanity_T5 or test_sanity_amp_and_nvfuser or test_sanity_drop_path or test_sanity_fused_qkv_params or test_sanity_gradient_accumulation_fusion or test_inference_mode or test_sanity_normalization_amp or test_sanity_layernorm_linear or test_sanity_linear_with_zero_tokens or test_sanity_grouped_linear)\" --no-header"
else
    # CUDA attention backward segfaults in these TransformerLayer sanity families.
    SANITY_CMD="NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py -k \"not (test_sanity_gpt or test_sanity_bert or test_sanity_T5 or test_sanity_amp_and_nvfuser or test_sanity_drop_path or test_sanity_fused_qkv_params or test_sanity_gradient_accumulation_fusion)\" --no-header"
fi
run_test_step "pytest_test_sanity.xml" "$TE_PATH/tests/pytorch/test_sanity.py" "$SANITY_CMD" "test_sanity.py"


# Step: Recipe
run_test_step "pytest_test_recipe.xml" "$TE_PATH/tests/pytorch/test_recipe.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py" "test_recipe.py"

# Step: Deferred Init
run_test_step "pytest_test_deferred_init.xml" "$TE_PATH/tests/pytorch/test_deferred_init.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py" "test_deferred_init.py"

# Step: Numerics
if [ "$PLATFORM" = "metax" ]; then
    NUMERICS_CMD="PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py -k \"not (test_layernorm_mlp_accuracy or test_grouped_linear_accuracy or test_gpt_cuda_graph or test_transformer_layer_hidden_states_format or test_grouped_gemm or test_noncontiguous or test_gpt_checkpointing or test_gpt_accuracy or test_mha_accuracy or test_linear_accuracy or test_linear_accuracy_delay_wgrad_compute or test_rmsnorm_accuracy or test_layernorm_accuracy or test_layernorm_linear_accuracy)\" --no-header"
else
    # CUDA
    NUMERICS_CMD="PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py -k \"not (test_linear_accuracy or test_layernorm_linear_accuracy or test_layernorm_mlp_accuracy or test_transformer_layer_hidden_states_format)\" --no-header"
fi
run_test_step "pytest_test_numerics.xml" "$TE_PATH/tests/pytorch/test_numerics.py" "$NUMERICS_CMD" "test_numerics.py"

# Step: CUDA Graphs
run_test_step "pytest_test_cuda_graphs.xml" "$TE_PATH/tests/pytorch/test_cuda_graphs.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py" "test_cuda_graphs.py"

# Step: JIT
if [ "$PLATFORM" = "metax" ]; then
    JIT_CMD="python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py -k \"not (test_torch_dynamo)\""
else
    JIT_CMD="python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py --no-header"
fi
run_test_step "pytest_test_jit.xml" "$TE_PATH/tests/pytorch/test_jit.py" "$JIT_CMD" "test_jit.py"


# Step: Fused Rope
run_test_step "pytest_test_fused_rope.xml" "$TE_PATH/tests/pytorch/test_fused_rope.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py" "test_fused_rope.py"

# Step: NVFP4 (Directory)
run_test_step "pytest_test_nvfp4.xml" "$TE_PATH/tests/pytorch/nvfp4" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4.xml $TE_PATH/tests/pytorch/nvfp4" "test_nvfp4"

# Step: Quantized Tensors
run_test_step "pytest_test_quantized_tensor.xml" "$TE_PATH/tests/pytorch/test_quantized_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_quantized_tensor.xml $TE_PATH/tests/pytorch/test_quantized_tensor.py" "test_quantized_tensor.py"

# Step: Float8 Blockwise Tensor
run_test_step "pytest_test_float8blockwisetensor.xml" "$TE_PATH/tests/pytorch/test_float8blockwisetensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py" "test_float8blockwisetensor.py"

# Step: Float8 Blockwise Scaling Exact
run_test_step "pytest_test_float8_blockwise_scaling_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" "test_float8_blockwise_scaling_exact.py"

# Step: Float8 Blockwise GEMM Exact
run_test_step "pytest_test_float8_blockwise_gemm_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" "test_float8_blockwise_gemm_exact.py"

# Step: GQA
run_test_step "pytest_test_gqa.xml" "$TE_PATH/tests/pytorch/test_gqa.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py" "test_gqa.py"

# Step: Fused Optimizer
run_test_step "pytest_test_fused_optimizer.xml" "$TE_PATH/tests/pytorch/test_fused_optimizer.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py" "test_fused_optimizer.py"

# Step: Multi Tensor
run_test_step "pytest_test_multi_tensor.xml" "$TE_PATH/tests/pytorch/test_multi_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_multi_tensor.xml $TE_PATH/tests/pytorch/test_multi_tensor.py" "test_multi_tensor.py"

# Step: Fusible Ops
run_test_step "pytest_test_fusible_ops.xml" "$TE_PATH/tests/pytorch/test_fusible_ops.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/test_fusible_ops.py" "test_fusible_ops.py"

# Step: Permutation
run_test_step "pytest_test_permutation.xml" "$TE_PATH/tests/pytorch/test_permutation.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_permutation.xml $TE_PATH/tests/pytorch/test_permutation.py" "test_permutation.py"

# Step: Cross Entropy
run_test_step "pytest_test_cross_entropy.xml" "$TE_PATH/tests/pytorch/test_cross_entropy.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cross_entropy.xml $TE_PATH/tests/pytorch/test_cross_entropy.py" "test_cross_entropy.py"

# Step: CPU Offloading
run_test_step "pytest_test_cpu_offloading.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py" "test_cpu_offloading.py"

# Step: CPU Offloading V1
run_test_step "pytest_test_cpu_offloading_v1.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" \
"NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading_v1.xml $TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" "test_cpu_offloading_v1.py"

# Step: Attention
run_test_step "pytest_test_attention.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" \
"NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention.xml $TE_PATH/tests/pytorch/attention/test_attention.py" "test_attention.py"

# Step: KV Cache
run_test_step "pytest_test_kv_cache.xml" "$TE_PATH/tests/pytorch/attention/test_kv_cache.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_kv_cache.xml $TE_PATH/tests/pytorch/attention/test_kv_cache.py" "test_kv_cache.py"

# Step: HF Integration
run_test_step "pytest_test_hf_integration.xml" "$TE_PATH/tests/pytorch/test_hf_integration.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hf_integration.xml $TE_PATH/tests/pytorch/test_hf_integration.py" "test_hf_integration.py"

# Step: Checkpoint
export NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint
# Generate fixtures only on platforms where the original FL driver runs this group.
if [ "$PLATFORM" != "metax" ] && [[ "$IS_CUDA_BACKEND" != *"cuda"* ]] && [ ! -d "$NVTE_TEST_CHECKPOINT_ARTIFACT_PATH" ]; then
    python3 $TE_PATH/tests/pytorch/test_checkpoint.py --save-checkpoint all || exit 1
fi
run_test_step "pytest_test_checkpoint.xml" "$TE_PATH/tests/pytorch/test_checkpoint.py" \
"NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_checkpoint.xml $TE_PATH/tests/pytorch/test_checkpoint.py" "test_checkpoint.py"

# ==============================================================================
# New Step: Plugin Core
# ==============================================================================
PLUGIN_TEST_ROOT="$TE_PATH/tests/plugin"

# Step: Plugin Policy
run_test_step "pytest_test_plugin_policy.xml" "$PLUGIN_TEST_ROOT/plugin/test_policy.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_policy.xml $PLUGIN_TEST_ROOT/plugin/test_policy.py" "test_policy.py"

# Step: Plugin manager
run_test_step "pytest_test_plugin_manager.xml" "$PLUGIN_TEST_ROOT/plugin/test_manager.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_manager.xml $PLUGIN_TEST_ROOT/plugin/test_manager.py" "test_manager.py"

# Step: Plugin policy selection
run_test_step "pytest_test_plugin_policy_selection.xml" "$PLUGIN_TEST_ROOT/plugin/test_policy_selection.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_plugin_policy_selection.xml $PLUGIN_TEST_ROOT/plugin/test_policy_selection.py" "test_policy_selection.py"


# ==============================================================================
# New Step: Plugin Core backend
# ==============================================================================

# Step: Backend flagos =========================================================
run_test_step "pytest_test_backend_flagos.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_lifecycle.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos.xml $PLUGIN_TEST_ROOT/backend/flagos/test_lifecycle.py" "test_lifecycle.py"

# Step: Backend impl fused rope
run_test_step "pytest_test_backend_flagos_fused_rope.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_fused_rope.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_fused_rope.xml $PLUGIN_TEST_ROOT/backend/flagos/test_fused_rope.py" "test_fused_rope.py"

# Step: Backend impl fused adam
run_test_step "pytest_test_backend_flagos_fused_adam.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_optimizer.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_fused_adam.xml $PLUGIN_TEST_ROOT/backend/flagos/test_optimizer.py" "test_optimizer.py"

# Step: Backend impl gemm
run_test_step "pytest_test_backend_flagos_gemm.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_gemm.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_gemm.xml $PLUGIN_TEST_ROOT/backend/flagos/test_gemm.py" "test_gemm.py"

# Step: Backend impl multi_tensor
run_test_step "pytest_test_backend_flagos_multi_tensor.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_multi_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_multi_tensor.xml $PLUGIN_TEST_ROOT/backend/flagos/test_multi_tensor.py" "test_multi_tensor.py"

# Step: Backend impl rmsnorm
run_test_step "pytest_test_backend_flagos_rmsnorm.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_rmsnorm.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_rmsnorm.xml $PLUGIN_TEST_ROOT/backend/flagos/test_rmsnorm.py" "test_rmsnorm.py"

# Step: Backend impl softmax
run_test_step "pytest_test_backend_flagos_softmax.xml" "$PLUGIN_TEST_ROOT/backend/flagos/test_softmax.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_flagos_softmax.xml $PLUGIN_TEST_ROOT/backend/flagos/test_softmax.py" "test_softmax.py"


# Step: Backend reference =========================================================
run_test_step "pytest_test_backend_reference.xml" "$PLUGIN_TEST_ROOT/backend/reference/test_lifecycle.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference.xml $PLUGIN_TEST_ROOT/backend/reference/test_lifecycle.py" "test_lifecycle.py"

run_test_step "pytest_test_backend_reference_activation.xml" "$PLUGIN_TEST_ROOT/backend/reference/test_activation.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_activation.xml $PLUGIN_TEST_ROOT/backend/reference/test_activation.py" "test_activation.py"

run_test_step "pytest_test_backend_reference_dropout.xml" "$PLUGIN_TEST_ROOT/backend/reference/test_dropout.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_dropout.xml $PLUGIN_TEST_ROOT/backend/reference/test_dropout.py" "test_dropout.py"

run_test_step "pytest_test_backend_reference_gemm.xml" "$PLUGIN_TEST_ROOT/backend/reference/test_gemm.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backend_reference_gemm.xml $PLUGIN_TEST_ROOT/backend/reference/test_gemm.py" "test_gemm.py"


# Additional upstream coverage and FL upgrade regression tests.
run_test_step "pytest_test_custom_recipe.xml" "$TE_PATH/tests/pytorch/test_custom_recipe.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_custom_recipe.xml $TE_PATH/tests/pytorch/test_custom_recipe.py" "test_custom_recipe.py"

run_test_step "pytest_test_mxfp8.xml" "$TE_PATH/tests/pytorch/mxfp8" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8.xml $TE_PATH/tests/pytorch/mxfp8" "test_mxfp8"

run_test_step "pytest_test_weight_swizzle_in_layers.xml" "$TE_PATH/tests/pytorch/test_weight_swizzle_in_layers.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_weight_swizzle_in_layers.xml $TE_PATH/tests/pytorch/test_weight_swizzle_in_layers.py" "test_weight_swizzle_in_layers.py"

run_test_step "pytest_test_torch_compile.xml" "$TE_PATH/tests/pytorch/test_torch_compile.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_torch_compile.xml $TE_PATH/tests/pytorch/test_torch_compile.py" "test_torch_compile.py"

run_test_step "pytest_test_float8_current_scaling_exact.xml" "$TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_current_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py" "test_float8_current_scaling_exact.py"

run_test_step "test_grouped_tensor.xml" "$TE_PATH/tests/pytorch/test_grouped_tensor.py" \
"NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_grouped_tensor.xml $TE_PATH/tests/pytorch/test_grouped_tensor.py" "test_grouped_tensor.py"

run_test_step "pytest_test_qk_norm.xml" "$TE_PATH/tests/pytorch/test_qk_norm.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_qk_norm.xml $TE_PATH/tests/pytorch/test_qk_norm.py" "test_qk_norm.py"

run_test_step "pytest_test_selective_activation_checkpoint.xml" "$TE_PATH/tests/pytorch/layernorm_mlp/test_selective_activation_checkpoint.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_selective_activation_checkpoint.xml $TE_PATH/tests/pytorch/layernorm_mlp/test_selective_activation_checkpoint.py" "test_selective_activation_checkpoint.py"

run_test_step "pytest_test_distributed_weight.xml" "$TE_PATH/tests/pytorch/test_distributed_weight.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_distributed_weight.xml $TE_PATH/tests/pytorch/test_distributed_weight.py" "test_distributed_weight.py"

run_test_step "pytest_test_backward_override.xml" "$TE_PATH/tests/pytorch/test_backward_override.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backward_override.xml $TE_PATH/tests/pytorch/test_backward_override.py" "test_backward_override.py"

run_test_step "pytest_test_hybrid_quantization.xml" "$TE_PATH/tests/pytorch/test_hybrid_quantization.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hybrid_quantization.xml $TE_PATH/tests/pytorch/test_hybrid_quantization.py" "test_hybrid_quantization.py"

run_test_step "pytest_test_identity_quantizer.xml" "$TE_PATH/tests/pytorch/test_identity_quantizer.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_identity_quantizer.xml $TE_PATH/tests/pytorch/test_identity_quantizer.py" "test_identity_quantizer.py"

run_test_step "pytest_test_flex_attention.xml" "$TE_PATH/tests/pytorch/attention/test_flex_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_flex_attention.xml $TE_PATH/tests/pytorch/attention/test_flex_attention.py" "test_flex_attention.py"

# Respect upstream capability skips on vendor backends. Dedicated CUDA CI may
# opt into strict dependency checks through NVTE_{GDN,GDN2,GDP}_TEST_REQUIRED.
run_test_step "pytest_test_gdn_attention.xml" "$TE_PATH/tests/pytorch/attention/test_gdn_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdn_attention.xml $TE_PATH/tests/pytorch/attention/test_gdn_attention.py" "test_gdn_attention.py"

run_test_step "pytest_test_gdn2_attention.xml" "$TE_PATH/tests/pytorch/attention/test_gdn2_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdn2_attention.xml $TE_PATH/tests/pytorch/attention/test_gdn2_attention.py" "test_gdn2_attention.py"

run_test_step "pytest_test_gdp_attention.xml" "$TE_PATH/tests/pytorch/attention/test_gdp_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gdp_attention.xml $TE_PATH/tests/pytorch/attention/test_gdp_attention.py" "test_gdp_attention.py"

run_test_step "pytest_test_attention_deterministic.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" \
"NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention_deterministic.xml $TE_PATH/tests/pytorch/attention/test_attention.py" "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_attention.py"

run_test_step "pytest_test_linear_mxfp8_attention.xml" "$TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_linear_mxfp8_attention.xml $TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py" "test_linear_mxfp8_attention.py"

run_test_step "pytest_test_fused_mla_q_uproj.xml" "$TE_PATH/tests/pytorch/attention/test_fused_mla_q_uproj.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_mla_q_uproj.xml $TE_PATH/tests/pytorch/attention/test_fused_mla_q_uproj.py" "test_fused_mla_q_uproj.py"

run_test_step "pytest_test_cu_seqlens_cache.xml" "$TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cu_seqlens_cache.xml $TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py" "test_cu_seqlens_cache.py"

run_test_step "pytest_test_mixed_thd_attention.xml" "$TE_PATH/tests/pytorch/attention/test_mixed_thd_attention.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mixed_thd_attention.xml $TE_PATH/tests/pytorch/attention/test_mixed_thd_attention.py" "test_mixed_thd_attention.py"

run_test_step "pytest_test_fused_router.xml" "$TE_PATH/tests/pytorch/test_fused_router.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_router.xml $TE_PATH/tests/pytorch/test_fused_router.py" "test_fused_router.py"

run_test_step "pytest_test_partial_cast.xml" "$TE_PATH/tests/pytorch/test_partial_cast.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_partial_cast.xml $TE_PATH/tests/pytorch/test_partial_cast.py" "test_partial_cast.py"

run_test_step "pytest_test_mxfp8_2d_quantize.xml" "$TE_PATH/tests/pytorch/test_mxfp8_2d_quantize.py" \
"python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8_2d_quantize.xml $TE_PATH/tests/pytorch/test_mxfp8_2d_quantize.py" "test_mxfp8_2d_quantize.py"

run_test_step "pytest_test_mhc.xml" "$TE_PATH/tests/pytorch/test_mhc.py" \
"NVTE_DISABLE_TRITON_AUTOTUNING=1 NVIDIA_TF32_OVERRIDE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mhc.xml $TE_PATH/tests/pytorch/test_mhc.py" "test_mhc.py"

run_test_step "pytest_test_mhc_deterministic.xml" "$TE_PATH/tests/pytorch/test_mhc.py" \
"NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_DISABLE_TRITON_AUTOTUNING=1 NVIDIA_TF32_OVERRIDE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mhc_deterministic.xml $TE_PATH/tests/pytorch/test_mhc.py" "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_mhc.py"

run_test_step "pytest_test_grouped_linear.xml" "$TE_PATH/tests/pytorch/test_grouped_linear.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_linear.xml $TE_PATH/tests/pytorch/test_grouped_linear.py" "test_grouped_linear.py"

run_test_step "pytest_test_ops_grouped_linear_distributed_weight.xml" "$TE_PATH/tests/pytorch/test_ops_grouped_linear_distributed_weight.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_ops_grouped_linear_distributed_weight.xml $TE_PATH/tests/pytorch/test_ops_grouped_linear_distributed_weight.py" "test_ops_grouped_linear_distributed_weight.py"

run_test_step "pytest_test_grouped_mlp.xml" "$TE_PATH/tests/pytorch/test_grouped_mlp.py" \
"NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_mlp.xml $TE_PATH/tests/pytorch/test_grouped_mlp.py" "test_grouped_mlp.py"

# FL upgrade contracts remain in the original plugin test driver.
run_test_step "pytest_v220_api.xml" "$PLUGIN_TEST_ROOT/plugin/test_v220_api_contract.py" \
"python3 -m pytest --junitxml=$XML_LOG_DIR/pytest_v220_api.xml $PLUGIN_TEST_ROOT/plugin/test_v220_api_contract.py $PLUGIN_TEST_ROOT/plugin/test_vendor_compat.py $PLUGIN_TEST_ROOT/plugin/test_device_helpers.py" "v2.20.2 API contract"

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
echo "All assigned tests passed (some might have been skipped)."
exit 0
