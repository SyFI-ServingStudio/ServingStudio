# Kimi-K3 end-to-end serving benchmark — all optimizations vs stock sglang (8 × B200, 2026-10-13/14)

*Companion to `K3_OPTIMIZATION_REPORT.md` (single-layer results, techniques, PR plan). This document is the full record
of the end-to-end runs: what model was served, the exact server and client commands, every workload, every run.*

## 1. Summary

All accepted KDA and MLA optimizations, merged into one sglang tree, make end-to-end serving of the Kimi-K3
architecture faster than stock sglang v0.5.20 on every workload tested (medians of 3 runs each):

| workload | headline metric | stock | all optimizations | Δ |
|---|---|---|---|---|
| W1 one 1,040,000-token request → 32 tokens | TTFT (= the 1M-token prefill) | 54.5 s | 45.3 s | **−16.8%** |
| | throughput (input tok/s) | 19,007 | 22,835 | **+20.1%** |
| | TPOT (decode at 1M context) | 6.13 ms | 5.70 ms | −7.0% |
| W2 128 × 8k → 256 | output tok/s | 1,262 | 1,371 | +8.7% |
| W4 16 × 16k prefill-only | input tok/s | 77,184 | 105,976 | **+37.3%** (vs the best stock run: +11–18%) |
| | median TTFT | 731 ms | 645 ms | **−11.8%** |
| W5 512 × 8k → 128 | output tok/s | 705 | 848 | **+20.2%** |
| W6 16 × 64k → 64 | output tok/s | 42.3 | 46.7 | **+10.4%** |
| W3 mixed prefill+decode, `--enable-mixed-chunk` (1 run) | TPOT / output tok/s | 59.9 ms / 744 | 32.2 ms / 1,027 | **−46.3% / +38.1%** |
| W2 with `--enable-mixed-chunk` (1 run) | output tok/s | 909 | 1,352 | **+48.8%** |

Correctness of the same tree: all 8 single-layer judge cases pass against stock sglang's outputs (max rel error ≤
0.0185, limit 0.02), §7.

**What this is and is not.** It is a real `sglang.launch_server` + `sglang.bench_serving` run of the Kimi-K3
architecture on one 8-GPU node, with **48 of K3's 93 layers and dummy (random) weights**. It is not the full model and
it measures speed only, not accuracy (§2, §8).

## 2. Why a 48-layer K3 with dummy weights

- **No checkpoint on the node.** `moonshotai/Kimi-K3` is ~1.5 TB of safetensors; `/raid` has 1.3 TB free. Only the
  config, tokenizer and model code were downloaded (`/raid/yilegu/models/kimi-k3-meta`).
- **The full model does not fit one 8 × B200 node.** Routed experts alone: 896 experts × 92 MoE layers × (3 × 3584 ×
  3072) ≈ 2.7 T parameters ≈ 1.44 TB in MXFP4, versus 8 × 183 GB ≈ 1.46 TB of HBM before attention weights, KV cache
  and activations. Every B200 recipe in the sglang Kimi-K3 cookbook uses **2 nodes × 8 GPUs** (PP2 × TP8 or PP2 ×
  DCPEP8); only the 288 GB B300 has a 1 × 8 recipe.
- **48 layers ≈ one pipeline stage of the B200 PP2 × TP8 recipe**, and it fits: ~92 GB of MoE weights per GPU at EP8,
  leaving room for a 4.06 M-token fp8 KV pool and the KDA state pool.

## 3. The model that was served

`/raid/yilegu/models/kimi-k3-L48-dummy/` = the real files from `moonshotai/Kimi-K3` (`config.json`,
`configuration_kimi_k3.py`, `modeling_kimi_k3.py`, `modeling_kimi_linear.py`, `tiktoken.model`, tokenizer and
processor files) with exactly one change in `config.json` → `text_config`:

| field | real K3 | served |
|---|---|---|
| `num_hidden_layers` | 93 | 48 |
| `linear_attn_config.full_attn_layers` (MLA, 1-based) | 24 layers: 4, 8, …, 92, 93 | 12 layers: 4, 8, …, 48 |
| `linear_attn_config.kda_layers` | 69 layers | 36 layers (all others in 1–48) |
| first dense layer (`first_k_dense_replace`) | 1 | 1 |

Everything else is unchanged: hidden 7168; 96 attention heads (12 per GPU at TP8); MLA q_lora 1536, kv_lora 512, rope
64, output gate; KDA head_dim 128, gate lower bound −5, short conv 4; 896 routed experts, top-16, 2 shared experts,
latent MoE (routed hidden 3584, intermediate 3072), SiTU activation; attention residuals; vocab 163,840; MXFP4
(compressed-tensors, group 32) routed experts, bf16 elsewhere; vision tower present but unused. Weights are created by
`--load-format dummy` (random values), so expert routing is close to uniform rather than the trained distribution.

## 4. Server (identical for every run except the tree)

One slurm job per run (`k3_e2e/k3_e2e_bench.sbatch`, partition `main`, 8 GPUs, `--time 00:45:00`; each run took 9–12
minutes including ~7 min of server start-up). Image `lmsysorg/sglang:v0.5.20`; the tree under test is bind-mounted over
`/sgl-workspace/sglang/python/sglang` (stock runs mount nothing); `__pycache__` is cleared before start.

```bash
python3 -m sglang.launch_server --model-path /models/k3 --load-format dummy --trust-remote-code \
  --tp-size 8 --ep-size 8 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.85 --context-length 1048576 --chunked-prefill-size 16384 \
  --max-running-requests 160 --watchdog-timeout 3600 --host 127.0.0.1 --port <20000+jobid%10000> \
  [--enable-mixed-chunk]        # only for the two mixed-chunk runs
```

Resolved by sglang at start-up (from the server info of the stock run): attention backend `trtllm_mla` for prefill and
decode (K3's SM100 default), MoE runner `flashinfer_mxfp4`, no all-to-all backend (EP via all-reduce), PP 1, DP 1, page
size 64, CUDA graphs on, FlashInfer autotune on, radix cache on, FCFS scheduling, `mamba_full_memory_ratio` 0.9.
Memory per GPU: fp8 KV pool 4,061,248 tokens (26.1 GB), KDA state pool 1,666 requests (conv 1.55 GB + ssm 22.0 GB),
~23.5 GB free after start-up.

## 5. Client and workloads

`sglang.bench_serving` run inside the server container against `127.0.0.1`:

```bash
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port <port> --tokenizer /models/k3 \
  --dataset-name random-ids --tokenize-prompt --random-input-len <IN> --random-output-len <OUT> \
  --random-range-ratio 1.0 --num-prompts <N> --max-concurrency <C> --request-rate <R> \
  --warmup-requests <WU> --output-file <workload>.jsonl
```

`random-ids` + `--tokenize-prompt` sends random token ids, so every request has exactly IN input tokens (without
`--tokenize-prompt` the 1M prompt re-tokenized to 1,063,379 tokens and the server rejected it while the client still
reported success — the script now marks a workload invalid if the server logs any rejection; no measured run has one).
`--random-range-ratio 1.0` = fixed lengths; outputs ignore EOS, so every request generates exactly OUT tokens. The
prompts are random, so the radix cache never hits. Each job first runs a `smoke` workload (2 × 8192 → 64, no warmup)
that absorbs JIT compilation; its numbers are not reported.

| id | N prompts | IN tokens | OUT tokens | max concurrency C | request rate R | warmup | what it exercises |
|---|---|---|---|---|---|---|---|
| W1 | 1 | 1,040,000 | 32 | 1 | all at once | 0 | one 1M-context request: TTFT = chunked prefill of 1.04 M tokens (64 chunks of 16k); TPOT = decode at ~1M context |
| W2 | 128 | 8,192 | 256 | 128 | all at once | 1 | throughput decode at batch 128 (after a prefill burst) |
| W3 | 96 | 16,384 | 512 | 64 | 2 req/s | 1 | arrivals while others decode → mixed prefill+decode batches (run on the `--enable-mixed-chunk` server) |
| W4 | 16 | 16,384 | 1 | 4 | all at once | 1 | prefill throughput, 16k prompts, no prefix |
| W5 | 512 | 8,192 | 128 | 512 | all at once | 1 | large batch 512 × 8k (decode at batch up to 160 = `--max-running-requests`, the rest queue) |
| W6 | 16 | 65,536 | 64 | 16 | all at once | 0 | small batch at 64k context |

Metrics are bench_serving's: TTFT (request sent → first token), TPOT (mean time per output token after the first),
E2E latency, and input / output / total token throughput over the benchmark duration.

## 6. Trees and runs

| label | tree | what is in it | jobs |
|---|---|---|---|
| stock | none (image's sglang v0.5.20) | — | 6794, 6806, 6808 |
| ALL | `/raid/yilegu/k3_e2e_trees/ALL_c8d52d4` = `/raid/yilegu/k3_merge` branch `all`, commit `c8d52d4` | every accepted lever of the MLA decode/b512/shapes and MLA prefill/long-context/mixed lineages and the KDA decode/b512/shapes/verify and KDA prefill/long-context/mixed lineages (T1–T44, T49 of the report); where both lineages changed shared code (attn-res kernels, MoE front split, shared-expert overlap, SiTU) the variant is selected by layer type; MLA verify tree excluded; MLA b512 r5, MLA prefill r4/r6, KDA prefill r4 not included | 6797, 6807, 6809 |
| stock + mixed chunk | none, `--enable-mixed-chunk` | — | 6810 |
| ALL + mixed chunk | ALL, `--enable-mixed-chunk` | — | 6811 |
| B | stock + `mla_prefill_02_r2_prefix_attention_cutedsl_fmha.patch` + `mla_lcprefill_01_r1.patch` | candidate PR B (CuTe-DSL prefix FMHA + 16-byte alignment + KV split) | 6795 |
| D | stock + `kda/02_r22_bf16_state_fused_kda_decode_kernel.patch` | candidate PR D (bf16-state fused KDA decode) | 6796 |

Stock and ALL runs were interleaved (stock, ALL, stock, ALL, …) on the same node and GPUs.

## 7. Correctness of the ALL tree

The end-to-end runs use dummy weights, so correctness was checked with the single-layer judge (real sglang layers,
seeded weights, outputs compared to stock sglang on the same inputs, seed 1, fresh goldens): all 8 cases PASS and the
per-layer gains match the separate best trees.

| case | points | per-layer Δ vs stock | max rel error |
|---|---|---|---|
| KDA decode shapes | 1×1M / 16×64k / 128×8k / 32×8k / 1×8k | −26.2 / −16.0 / −7.7 / −14.6 / −25.4% | 0.0185 |
| MLA decode shapes | same | −32.9 / −14.4 / −6.6 / −10.0 / −18.1% | 0.0089 |
| MLA large batch | 512×8k / 256×8k / 16×64k mix / 128×8k | −12.3 / −6.7 / −13.3 / −6.6% | 0.0116 |
| KDA long-context prefill | 16k@245k / 32k@229k / 16k@131k / 32k@131k | −8.5 / −8.0 / −10.1 / −8.9% | 0.0074 |
| MLA long-context prefill | same | −26.5 / −17.1 / −19.2 / −14.3% | 0.0054 |
| KDA mixed | 64 dec + 16k / 128 dec + 4k / 64 dec + 16k@48k | −17.5 / −20.1 / −17.1% | 0.0072 |
| MLA mixed | same | −61.5 / −89.9 / −71.2% | 0.0113 |
| KDA verify | 64×4 / 128×4 / 16×4@64k | −6.8 / −8.9 / −9.4% | 0.0170 |

(Job 6814, sequential, plus job 6800 for KDA long-context prefill. A first attempt that ran 8 judges at once on one shared
tree directory failed 7 of 8 on `__pycache__`/JIT-build races; those verdicts are kept as `*.racefail.json`.)

## 8. Results (every run)

Δ = median(ALL) / median(stock) − 1; negative = faster for latencies, positive = better for throughputs.

### Default server (no mixed chunk): stock (jobs 6794, 6806, 6808) vs ALL (jobs 6797, 6807, 6809)

**W1 1 × 1,040,000 → 32**

| metric | stock runs | stock median | ALL runs | ALL median | Δ median |
|---|---|---|---|---|---|
| mean TTFT (ms) | 54,356.7 / 54,568.2 / 54,507.1 | 54,507.1 | 45,269.7 / 45,348.9 / 45,342.4 | 45,342.4 | -16.8% |
| median TTFT (ms) | 54,356.7 / 54,568.2 / 54,507.1 | 54,507.1 | 45,269.7 / 45,348.9 / 45,342.4 | 45,342.4 | -16.8% |
| P99 TTFT (ms) | 54,356.7 / 54,568.2 / 54,507.1 | 54,507.1 | 45,269.7 / 45,348.9 / 45,342.4 | 45,342.4 | -16.8% |
| mean TPOT (ms) | 6.128 / 6.534 / 6.117 | 6.128 | 5.700 / 5.497 / 6.042 | 5.700 | -7.0% |
| P99 TPOT (ms) | 6.128 / 6.534 / 6.117 | 6.128 | 5.700 / 5.497 / 6.042 | 5.700 | -7.0% |
| mean E2E latency (ms) | 54,546.6 / 54,770.7 / 54,696.8 | 54,696.8 | 45,446.4 / 45,519.3 / 45,529.7 | 45,519.3 | -16.8% |
| input tok/s | 19,056.2 / 18,980.3 / 19,006.6 | 19,006.6 | 22,871.3 / 22,834.8 / 22,829.9 | 22,834.8 | +20.1% |
| output tok/s | 0.586 / 0.584 / 0.585 | 0.585 | 0.704 / 0.703 / 0.702 | 0.703 | +20.1% |
| total tok/s | 19,056.8 / 18,980.9 / 19,007.2 | 19,007.2 | 22,872.0 / 22,835.5 / 22,830.6 | 22,835.5 | +20.1% |
| req/s | 0.018 / 0.018 / 0.018 | 0.018 | 0.022 / 0.022 / 0.022 | 0.022 | +20.1% |
| duration (s) | 54.6 / 54.8 / 54.7 | 54.7 | 45.5 / 45.5 / 45.6 | 45.5 | -16.8% |

**W2 128 × 8192 → 256**

| metric | stock runs | stock median | ALL runs | ALL median | Δ median |
|---|---|---|---|---|---|
| mean TTFT (ms) | 12,431.5 / 13,063.0 / 12,382.6 | 12,431.5 | 11,639.2 / 11,270.3 / 11,656.7 | 11,639.2 | -6.4% |
| median TTFT (ms) | 12,456.0 / 13,128.1 / 12,441.0 | 12,456.0 | 11,714.4 / 11,321.7 / 11,704.2 | 11,704.2 | -6.0% |
| P99 TTFT (ms) | 22,922.4 / 23,073.1 / 22,269.3 | 22,922.4 | 20,996.0 / 20,717.7 / 21,020.1 | 20,996.0 | -8.4% |
| mean TPOT (ms) | 52.7 / 50.7 / 50.4 | 50.7 | 47.7 / 47.9 / 47.8 | 47.8 | -5.8% |
| P99 TPOT (ms) | 95.0 / 94.3 / 92.9 | 94.3 | 88.2 / 88.0 / 87.9 | 88.0 | -6.7% |
| mean E2E latency (ms) | 25,878.4 / 25,987.9 / 25,226.0 | 25,878.4 | 23,808.2 / 23,489.1 / 23,835.4 | 23,808.2 | -8.0% |
| input tok/s | 40,382.4 / 40,201.3 / 41,418.4 | 40,382.4 | 43,885.4 / 44,478.6 / 43,827.2 | 43,885.4 | +8.7% |
| output tok/s | 1,262.0 / 1,256.3 / 1,294.3 | 1,262.0 | 1,371.4 / 1,390.0 / 1,369.6 | 1,371.4 | +8.7% |
| total tok/s | 41,644.4 / 41,457.5 / 42,712.8 | 41,644.4 | 45,256.8 / 45,868.5 / 45,196.8 | 45,256.8 | +8.7% |
| req/s | 4.929 / 4.907 / 5.056 | 4.929 | 5.357 / 5.430 / 5.350 | 5.357 | +8.7% |
| duration (s) | 26.0 / 26.1 / 25.3 | 26.0 | 23.9 / 23.6 / 23.9 | 23.9 | -8.0% |

**W4 16 × 16384 → 1**

| metric | stock runs | stock median | ALL runs | ALL median | Δ median |
|---|---|---|---|---|---|
| mean TTFT (ms) | 711.9 / 831.0 / 887.5 | 831.0 | 620.3 / 586.0 / 588.9 | 588.9 | -29.1% |
| median TTFT (ms) | 723.5 / 730.9 / 736.8 | 730.9 | 653.1 / 644.8 / 645.0 | 645.0 | -11.8% |
| P99 TTFT (ms) | 881.1 / 1,506.1 / 1,563.1 | 1,506.1 | 727.6 / 650.9 / 653.2 | 653.2 | -56.6% |
| mean E2E latency (ms) | 711.9 / 831.0 / 887.5 | 831.0 | 620.3 / 586.0 / 588.9 | 588.9 | -29.1% |
| input tok/s | 89,892.9 / 77,183.7 / 72,302.6 | 77,183.7 | 99,905.9 / 106,455.7 / 105,976.3 | 105,976.3 | +37.3% |
| output tok/s | 5.487 / 4.711 / 4.413 | 4.711 | 6.098 / 6.498 / 6.468 | 6.468 | +37.3% |
| total tok/s | 89,898.4 / 77,188.4 / 72,307.0 | 77,188.4 | 99,912.0 / 106,462.2 / 105,982.7 | 105,982.7 | +37.3% |
| req/s | 5.487 / 4.711 / 4.413 | 4.711 | 6.098 / 6.498 / 6.468 | 6.468 | +37.3% |
| duration (s) | 2.916 / 3.396 / 3.626 | 3.396 | 2.624 / 2.462 / 2.474 | 2.474 | -27.2% |

**W5 512 × 8192 → 128**

| metric | stock runs | stock median | ALL runs | ALL median | Δ median |
|---|---|---|---|---|---|
| mean TTFT (ms) | 46,615.9 / 37,369.6 / 46,081.3 | 46,081.3 | 34,549.9 / 43,079.0 / 34,396.7 | 34,549.9 | -25.0% |
| median TTFT (ms) | 46,434.6 / 35,913.2 / 45,876.0 | 45,876.0 | 33,090.2 / 42,917.1 / 32,961.9 | 33,090.2 | -27.9% |
| P99 TTFT (ms) | 91,744.7 / 81,236.6 / 91,161.6 | 91,161.6 | 75,610.1 / 85,482.6 / 75,442.6 | 75,610.1 | -17.1% |
| mean E2E latency (ms) | 60,187.3 / 49,676.6 / 59,692.6 | 59,692.6 | 46,172.5 / 56,006.8 / 45,993.6 | 46,172.5 | -22.6% |
| input tok/s | 44,871.7 / 50,599.6 / 45,125.5 | 45,125.5 | 54,258.3 / 48,079.4 / 54,445.2 | 54,258.3 | +20.2% |
| output tok/s | 701.1 / 790.6 / 705.1 | 705.1 | 847.8 / 751.2 / 850.7 | 847.8 | +20.2% |
| total tok/s | 45,572.9 / 51,390.2 / 45,830.6 | 45,830.6 | 55,106.1 / 48,830.7 / 55,296.0 | 55,106.1 | +20.2% |
| req/s | 5.478 / 6.177 / 5.508 | 5.508 | 6.623 / 5.869 / 6.646 | 6.623 | +20.2% |
| duration (s) | 93.5 / 82.9 / 92.9 | 92.9 | 77.3 / 87.2 / 77.0 | 77.3 | -16.8% |

**W6 16 × 65536 → 64**

| metric | stock runs | stock median | ALL runs | ALL median | Δ median |
|---|---|---|---|---|---|
| mean TTFT (ms) | 12,960.3 / 13,074.2 / 12,997.0 | 12,997.0 | 11,604.1 / 11,614.8 / 11,576.9 | 11,604.1 | -10.7% |
| median TTFT (ms) | 12,999.4 / 13,130.8 / 13,022.4 | 13,022.4 | 11,620.6 / 11,615.5 / 11,612.2 | 11,615.5 | -10.8% |
| P99 TTFT (ms) | 23,449.2 / 23,574.0 / 23,493.7 | 23,493.7 | 21,252.1 / 21,276.2 / 21,203.3 | 21,252.1 | -9.5% |
| mean TPOT (ms) | 176.6 / 176.9 / 177.1 | 176.9 | 163.0 / 163.5 / 162.8 | 163.0 | -7.8% |
| P99 TPOT (ms) | 348.2 / 349.7 / 349.4 | 349.4 | 319.7 / 320.2 / 319.1 | 319.7 | -8.5% |
| mean E2E latency (ms) | 24,086.7 / 24,216.8 / 24,156.2 | 24,156.2 | 21,875.6 / 21,916.0 / 21,833.7 | 21,875.6 | -9.4% |
| input tok/s | 43,449.6 / 43,213.9 / 43,302.8 | 43,302.8 | 47,821.3 / 47,742.7 / 47,925.2 | 47,821.3 | +10.4% |
| output tok/s | 42.4 / 42.2 / 42.3 | 42.3 | 46.7 / 46.6 / 46.8 | 46.7 | +10.4% |
| total tok/s | 43,492.0 / 43,256.1 / 43,345.1 | 43,345.1 | 47,868.0 / 47,789.4 / 47,972.0 | 47,868.0 | +10.4% |
| req/s | 0.663 / 0.659 / 0.661 | 0.661 | 0.730 / 0.728 / 0.731 | 0.730 | +10.4% |
| duration (s) | 24.1 / 24.3 / 24.2 | 24.2 | 21.9 / 22.0 / 21.9 | 21.9 | -9.4% |

### `--enable-mixed-chunk` server: stock (job 6810) vs ALL (job 6811), 1 run each

**W3 96 × 16384 → 512 at 2 req/s, max concurrency 64**

| metric | stock | ALL | Δ |
|---|---|---|---|
| mean TTFT (ms) | 2,886.1 | 1,216.8 | -57.8% |
| median TTFT (ms) | 2,760.6 | 1,168.9 | -57.7% |
| P99 TTFT (ms) | 5,845.9 | 2,308.8 | -60.5% |
| mean TPOT (ms) | 59.9 | 32.2 | -46.3% |
| P99 TPOT (ms) | 87.4 | 54.2 | -38.0% |
| mean E2E latency (ms) | 33,505.0 | 17,667.7 | -47.3% |
| input tok/s | 23,805.2 | 32,863.5 | +38.1% |
| output tok/s | 743.9 | 1,027.0 | +38.1% |
| total tok/s | 24,549.1 | 33,890.5 | +38.1% |
| req/s | 1.5 | 2.0 | +38.1% |
| duration (s) | 66.1 | 47.9 | -27.6% |

**W2 128 × 8192 → 256**

| metric | stock | ALL | Δ |
|---|---|---|---|
| mean TTFT (ms) | 15,889.3 | 11,357.2 | -28.5% |
| median TTFT (ms) | 14,899.7 | 11,319.9 | -24.0% |
| P99 TTFT (ms) | 33,046.8 | 21,365.9 | -35.3% |
| mean TPOT (ms) | 77.7 | 49.2 | -36.7% |
| P99 TPOT (ms) | 131.7 | 87.1 | -33.9% |
| mean E2E latency (ms) | 35,703.7 | 23,900.5 | -33.1% |
| input tok/s | 29,082.0 | 43,276.8 | +48.8% |
| output tok/s | 908.8 | 1,352.4 | +48.8% |
| total tok/s | 29,990.8 | 44,629.2 | +48.8% |
| req/s | 3.6 | 5.3 | +48.8% |
| duration (s) | 36.1 | 24.2 | -32.8% |

### Single first-wave patches on stock, default server, 1 run each (vs stock median)

| workload | metric | stock median | B: prefix FMHA + align (job 6795) | Δ | D: bf16 fused KDA decode (job 6796) | Δ |
|---|---|---|---|---|---|---|
| W1 | mean TTFT (ms) | 54,507.1 | 51,473.9 | -5.6% | 54,452.1 | -0.1% |
| W1 | mean TPOT (ms) | 6.1 | 6.7 | +8.9% | 6.0 | -2.8% |
| W1 | output tok/s | 0.6 | 0.6 | +5.8% | 0.6 | +0.1% |
| W1 | input tok/s | 19,006.6 | 20,114.1 | +5.8% | 19,026.5 | +0.1% |
| W2 | mean TTFT (ms) | 12,431.5 | 12,322.4 | -0.9% | 12,329.1 | -0.8% |
| W2 | mean TPOT (ms) | 50.7 | 50.2 | -1.0% | 51.8 | +2.2% |
| W2 | output tok/s | 1,262.0 | 1,299.9 | +3.0% | 1,278.8 | +1.3% |
| W2 | input tok/s | 40,382.4 | 41,598.2 | +3.0% | 40,920.3 | +1.3% |
| W4 | mean TTFT (ms) | 831.0 | 847.6 | +2.0% | 725.0 | -12.7% |
| W4 | input tok/s | 77,183.7 | 75,700.6 | -1.9% | 88,245.8 | +14.3% |
| W5 | mean TTFT (ms) | 46,081.3 | 45,882.9 | -0.4% | 45,825.7 | -0.6% |
| W5 | input tok/s | 45,125.5 | 45,260.8 | +0.3% | 45,429.0 | +0.7% |
| W6 | mean TTFT (ms) | 12,997.0 | 12,907.8 | -0.7% | 12,954.8 | -0.3% |
| W6 | mean TPOT (ms) | 176.9 | 175.8 | -0.6% | 175.6 | -0.7% |
| W6 | output tok/s | 42.3 | 42.6 | +0.8% | 42.6 | +0.6% |
| W6 | input tok/s | 43,302.8 | 43,634.4 | +0.8% | 43,571.4 | +0.6% |

## 9. Reading the results

- **Noise.** Stock run-to-run spread: < 1% on W1 and W6, 2–4% on TPOT at W2/W5, but 15–25% on TTFT for W4 and W5
  (queueing-dominated: when requests arrive all at once, TTFT depends on admission order). Every ≥ 10% headline above
  exceeds its spread, with one caveat: stock W4 got slower run by run (89.9k → 77.2k → 72.3k input tok/s; P99
  TTFT up to 1.5 s) while ALL stayed at 99.9–106.5k, so the +37% W4 median overstates the steady gain — against the
  best stock run ALL is +11–18%, and the median-TTFT gain (−11.8%) is the more robust W4 number. W5 TTFT (34–47 s
  across runs) should be read with its spread in mind. The two mixed-chunk
  runs are single runs.
- **Where the gains come from.** W1 TTFT and W4: the prefill levers (MLA CuTe-DSL prefix attention with the alignment
  fix and KV split, fused fp8 K/V pack, KDA CUDA chunk scan, strided KDA kernels, fused activations). W1 TPOT, W2, W5,
  W6: the decode levers (MLA B=1 KV split, KDA fused bf16 decode kernel, overlap and kernel fusions). W3 and mixed-chunk
  W2: the MLA mixed-batch dispatch fix — with `--enable-mixed-chunk` stock sglang is slower than without it (W2 TPOT
  77.7 vs 50.7 ms) because the decode rows of a mixed batch take the chunked-prefix MHA path; the fix removes that
  penalty (49.2 ms).
- **Smaller than the single-layer numbers**, as expected: end-to-end time also contains embedding, LM head, sampling,
  scheduling, TP all-reduces and EP routing, which the levers do not touch, and some levers are guarded to the
  single-rank path the harness modelled.
- **Single patches.** D alone: −0.5…−2.8% TPOT (small, as expected for one kernel). B alone regressed W4 TTFT by +19%
  and W1 TPOT by +9% — B is not ready as a stand-alone PR; the regression has to be understood first (ALL contains the
  same levers plus their follow-ups and does not regress).

## 10. Limitations

1. **48 of 93 layers.** Per-layer behaviour is the real one; the fixed per-step costs are amortised over half as many
   layers, so the relative gains of the full model should be similar or slightly larger. The cross-node pipeline hop
   of the real 16-GPU B200 deployment is not exercised.
2. **Dummy weights.** No accuracy measurement (gsm8k etc. need the checkpoint); routing is near-uniform rather than the
   trained expert-popularity distribution.
3. **One deployment shape** (TP8/EP8, no DCP, no DP attention, no speculative decoding, no PD disaggregation) and one
   GPU type.
4. **sglang v0.5.20.** Upstream main has moved ~1600 commits; numbers on main need re-measurement.

## 11. Reproduce

```bash
cd scripts-local/vibesim-analysis-container/k3_e2e
TREE=pristine LABEL=stock WORKLOADS="smoke w1 w2 w4 w5 w6" sbatch k3_e2e_bench.sbatch
TREE=/raid/yilegu/k3_e2e_trees/ALL_c8d52d4 LABEL=ALL WORKLOADS="smoke w1 w2 w4 w5 w6" sbatch k3_e2e_bench.sbatch
TREE=pristine LABEL=stock_mixedchunk EXTRA_SERVER_ARGS="--enable-mixed-chunk" WORKLOADS="smoke w3 w2" sbatch k3_e2e_bench.sbatch
# results: results/<LABEL>_<jobid>/{<workload>.jsonl,<workload>.txt,server.log,meta.txt}
# correctness: sbatch --partition=preemptible ../slurm_gpu.sh ../k3_slurm_step.sh ./mergecheck_seq.sh
```
