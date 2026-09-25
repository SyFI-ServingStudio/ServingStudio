# KDA trial 9: agent iterations

## iter_01
### result.json

```
{
  "sglang_version": "0.5.20",
  "moe_backend": "flashinfer_mxfp4",
  "hidden_act": "situ",
  "moe_method": {
    "class": "Mxfp4MoEMethod",
    "use_flashinfer": true,
    "use_marlin": false,
    "_fi_kernel": "trtllm_sm100",
    "flashinfer_mxfp4_moe_precision": "default"
  },
  "experts": 112,
  "ep": 8,
  "layer_idx": 5,
  "local_topk": 2,
  "hidden_scale": 1.0,
  "attn_type": "kda",
  "attention_backend": null,
  "page_size": null,
  "attn_heads": 12,
  "cuda_graph": true,
  "kv_cache_dtype": null,
  "mamba_ssm_dtype": "bfloat16",
  "num_attention_heads": 64,
  "config": {
    "num_experts": 112,
    "num_experts_per_token": 2,
    "num_shared_experts": 2,
    "moe_intermediate_size": 3072,
    "routed_expert_hidden_size": 3584,
    "hidden_size": 7168
  },
  "mla": null,
  "kda": {
    "num_heads": 12,
    "head_dim": 128,
    "short_conv_kernel": 4,
    "gate_lower_bound": -5.0
  },
  "points": [
    {
      "B": 128,
      "seq_len": 8192,
      "ok": true,
      "attn_type": "kda",
      "seed": 0,
      "attn_res": {
        "block_num": 8,
        "valid_blocks": 1,
        "write_layer": false
      },
      "out_shape": [
        128,
        7168
      ],
      "finite": true,
      "us_step": 1220.0640439987183,
      "us_token": 9.531750343739986,
      "state_mb": 54.291456,
      "peak_gb": 6.970517504,
      "graph_ok": true,
      "us_meta_prep": 36.54399886727333,
      "us_step_graph": 399.00800585746765,
      "graph_finite": true,
      "graph_speedup": 3.05774326852616,
      "us_token_graph": 3.117250045761466,
      "latency_us": 399.00800585746765,
      "latency_mode": "graph",
      "correctness": {
        "max_abs_err": 0.326171875,
        "max_rel_err": 0.0219275195342844,
        "mean_rel_err": 0.016420044004917145,
        "nan": false,
        "state": {
          "conv": {
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "mean_rel_err": 0.0,
            "nan": false
          },
          "temporal": {
            "max_abs_err": 0.00048828125,
            "max_rel_err": 0.0072991609585281525,
            "mean_rel_err": 0.00010127813584404066,
            "nan": false
          }
        },
        "state_ok": true,
        "rel_err_max": 0.02,
        "pass": false
      },
      "rung": "adhoc"
    },
    {
      "B": 32,
      "seq_len": 8192,
      "ok": true,
      "attn_type": "kda",
      "seed": 0,
      "attn_res": {
        "block_num": 8,
        "valid_blocks": 1,
        "writ
[... truncated]
```

## iter_02
### result.json

```
{
  "sglang_version": "0.5.20",
  "moe_backend": "flashinfer_mxfp4",
  "hidden_act": "situ",
  "moe_method": {
    "class": "Mxfp4MoEMethod",
    "use_flashinfer": true,
    "use_marlin": false,
    "_fi_kernel": "trtllm_sm100",
    "flashinfer_mxfp4_moe_precision": "default"
  },
  "experts": 112,
  "ep": 8,
  "layer_idx": 5,
  "local_topk": 2,
  "hidden_scale": 1.0,
  "attn_type": "kda",
  "attention_backend": null,
  "page_size": null,
  "attn_heads": 12,
  "cuda_graph": true,
  "kv_cache_dtype": null,
  "mamba_ssm_dtype": "bfloat16",
  "num_attention_heads": 64,
  "config": {
    "num_experts": 112,
    "num_experts_per_token": 2,
    "num_shared_experts": 2,
    "moe_intermediate_size": 3072,
    "routed_expert_hidden_size": 3584,
    "hidden_size": 7168
  },
  "mla": null,
  "kda": {
    "num_heads": 12,
    "head_dim": 128,
    "short_conv_kernel": 4,
    "gate_lower_bound": -5.0
  },
  "points": [
    {
      "B": 128,
      "seq_len": 8192,
      "ok": true,
      "attn_type": "kda",
      "seed": 0,
      "attn_res": {
        "block_num": 8,
        "valid_blocks": 1,
        "write_layer": false
      },
      "out_shape": [
        128,
        7168
      ],
      "finite": true,
      "us_step": 1242.5919771194458,
      "us_token": 9.70774982124567,
      "state_mb": 54.291456,
      "peak_gb": 6.970517504,
      "graph_ok": true,
      "us_meta_prep": 38.59199956059456,
      "us_step_graph": 410.97599267959595,
      "graph_finite": true,
      "graph_speedup": 3.023514753301398,
      "us_token_graph": 3.2107499428093433,
      "latency_us": 410.97599267959595,
      "latency_mode": "graph",
      "correctness": {
        "max_abs_err": 0.125,
        "max_rel_err": 0.008403360779605998,
        "mean_rel_err": 0.0012268954887986183,
        "nan": false,
        "state": {
          "conv": {
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "mean_rel_err": 0.0,
            "nan": false
          },
          "temporal": {
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "mean_rel_err": 0.0,
            "nan": false
          }
        },
        "state_ok": true,
        "rel_err_max": 0.02,
        "pass": true
      },
      "rung": "adhoc"
    },
    {
      "B": 32,
      "seq_len": 8192,
      "ok": true,
      "attn_type": "kda",
      "seed": 0,
      "attn_res": {
        "block_num": 8,
        "valid_blocks": 1,
        "write_layer": false
      },
      "out_shape": [
    
[... truncated]
```
