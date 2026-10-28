# Copyright (c) 2026, BAAI. All rights reserved.
# Shared policy for the FL L0/L1 PyTorch drivers. Source after setting TE_PATH,
# XML_LOG_DIR and FL_TEST_SUITE. Inherited exclusions remain visible in TSV.
: "${TE_FL_RUN_EXCLUDED:=0}"
FL_TEST_RESULTS="$XML_LOG_DIR/${FL_TEST_SUITE}_fl_results.tsv"
printf 'status\ttest\treason\n' > "$FL_TEST_RESULTS"
FL_CUDA_AVAILABLE=$(python3 -c 'import torch; print(int(torch.cuda.is_available()))') || error_exit "Cannot detect test device"

fl_skip_reason() {
    local test_path="$1"
    FL_SKIP_REASON=""
    [ "$TE_FL_RUN_EXCLUDED" = 1 ] && return 1
    if [ "$FL_TEST_SUITE" = L0 ]; then
        if [ "${PLATFORM:-}" = metax ]; then
            case "$test_path" in
                */test_numerics.py|*/test_sanity.py|*/test_cross_entropy.py|*/test_fused_rope.py|*/test_gqa.py|*/test_fused_optimizer.py|*/test_multi_tensor.py|*/test_cpu_offloading.py|*/test_cpu_offloading_v1.py|*/test_attention.py|*/attention/test_kv_cache.py|*/test_checkpoint.py|*/test_fused_router.py|*/test_cuda_graphs.py|*/test_hf_integration.py)
                    FL_SKIP_REASON="Inherited FL MetaX exclusion; requires validation on MetaX" ;;
            esac
        elif [ "$FL_CUDA_AVAILABLE" = 1 ]; then
            case "$test_path" in
                */test_checkpoint.py|*/test_cpu_offloading.py|*/test_cpu_offloading_v1.py|*/test_attention.py|*/attention/test_kv_cache.py|*/test_hf_integration.py)
                    FL_SKIP_REASON="Inherited FL CUDA exclusion; requires targeted revalidation" ;;
            esac
        fi
    else
        case "$test_path" in
            */distributed/test_sanity.py|*/distributed/test_fusible_ops.py|*/distributed/test_comm_gemm_overlap.py|*/distributed/test_fusible_ops_with_userbuffers.py|*/attention/test_attention_with_cp.py|*/debug/test_distributed.py)
                FL_SKIP_REASON="Disabled in prior FL distributed driver; capability validation pending" ;;
        esac
        if [ "${PLATFORM:-}" = metax ]; then
            case "$test_path" in
                */test_numerics.py|*/test_numerics_exact.py|*/test_torch_fsdp2.py|*/test_cast_master_weights_to_fp8.py)
                    FL_SKIP_REASON="Inherited FL MetaX distributed exclusion; requires validation on MetaX" ;;
            esac
        fi
    fi
    [ -n "$FL_SKIP_REASON" ]
}

run_test_step() {
    local test_path="$1" label="$2"
    shift 2
    if fl_skip_reason "$test_path"; then
        echo "[SKIP] $label: $FL_SKIP_REASON"
        printf 'skipped-by-policy\t%s\t%s\n' "$label" "$FL_SKIP_REASON" >> "$FL_TEST_RESULTS"
        return 0
    fi
    if [ "$FL_TEST_SUITE" = L1 ] && [ "$label" = "debug test_numerics.py" ] && [ "${DEBUG_TESTS_READY:-0}" != 1 ]; then
        echo "[BLOCKED] $label: nvdlfw_inspect unavailable"
        printf 'blocked\t%s\tnvdlfw_inspect unavailable\n' "$label" >> "$FL_TEST_RESULTS"
        return 0
    fi
    local selector=""
    if [ "$TE_FL_RUN_EXCLUDED" != 1 ] && [ "$FL_TEST_SUITE" = L0 ]; then
        case "$test_path" in
            */test_sanity.py) selector="not (test_sanity_gpt or test_sanity_bert or test_sanity_T5 or test_sanity_amp_and_nvfuser or test_sanity_drop_path or test_sanity_fused_qkv_params or test_sanity_gradient_accumulation_fusion)" ;;
            */test_numerics.py) selector="not (test_linear_accuracy or test_layernorm_linear_accuracy or test_layernorm_mlp_accuracy or test_transformer_layer_hidden_states_format)" ;;
            */test_jit.py) [ "${PLATFORM:-}" = metax ] && selector="not test_torch_dynamo" ;;
        esac
    fi
    if [ -n "$selector" ]; then
        printf 'filtered\t%s\tInherited FL selector: %s\n' "$label" "$selector" >> "$FL_TEST_RESULTS"
        set -- "$@" -k "$selector"
    fi
    echo "[RUN] $label"
    if "$@"; then
        printf 'command-passed\t%s\tSee JUnit for individual skips and deselections\n' "$label" >> "$FL_TEST_RESULTS"
    else
        printf 'failed\t%s\tCommand failed\n' "$label" >> "$FL_TEST_RESULTS"
        return 1
    fi
}
