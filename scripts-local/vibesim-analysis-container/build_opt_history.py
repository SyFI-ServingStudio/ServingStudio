#!/usr/bin/env python3
"""Build the Kimi-K3 optimization-history knowledge base (warm start for new workload cases).

Scans iter_opt_eval_k3_<case>/ (trial verdicts, judged patches, rounds.jsonl, the agents'
per-iteration hypothesis/analysis/result files) and writes k3_opt_history/:

  INDEX.md                      one row per judged trial/round: case, verdict, per-point deltas,
                                technique one-liner, files, links
  history.json                  the same, machine-readable
  TECHNIQUES.md                 curated levers + dead ends (hand-written, copied from
                                opt_history_techniques.md)
  trials/<case>_<k>/verdict.json
  trials/<case>_<k>/patch.diff          judged diff vs the PRISTINE tree (cumulative in the
                                        continuous loop)
  trials/<case>_<k>/incremental.diff    continuous rounds: diff vs the tree the round started
                                        from (= what THIS round changed)
  trials/<case>_<k>/agent_iterations.md every iteration's hypothesis + analysis + result
                                        (ideas tried inside the trial, incl. ones the agent
                                        itself rejected)

The directory is mounted read-only into the agent container (issue json "mounts") and the task
template tells the agent to consult it before forming hypotheses.
Usage: build_opt_history.py [--out k3_opt_history] [--cases kda,mla,...]
"""
import argparse, json, os, pathlib, re, shutil, subprocess, time

HERE = pathlib.Path(__file__).resolve().parent
SRC_EXT = (".py", ".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".inc", ".jinja")

# Hand-curated one-liners (what the trial changed / why it was rejected). Keyed by (case, k).
# Trials 1-3 were pipeline controls (null / planted slowdown / planted numerics), 4-7 ran on
# the campaign-1 workload with collapsed MoE routing (effect sizes unreliable, see TECHNIQUES).
NOTES = {
    ("kda", 1): "control: pristine tree (null)", ("kda", 2): "control: planted slowdown",
    ("kda", 3): "control: planted numerics", ("mla", 1): "control: pristine tree (null)",
    ("mla", 2): "control: planted slowdown", ("mla", 3): "control: planted numerics",
    ("kda", 4): "campaign 1: shared experts overlapped on the alt stream (+1.5%, bit-exact)",
    ("kda", 5): "campaign 1: forced the fused KDA decode kernel onto the bf16 state -> crashed at B=128",
    ("kda", 6): "campaign 1: best campaign-1 KDA result (-2.9% @128x8k)",
    ("kda", 7): "campaign 1: bf16 port of the fused KDA decode JIT kernel (.cuh); works, +1..3%, rel 0.017",
    ("kda", 8): "shared-expert down GEMM -> CuteDSL bf16 GEMM + side-stream overlap (-3.1%/-4.7%/-5.9%, bit-exact; 5% gate FAIL)",
    ("kda", 9): "589-line bf16 port of the fused KDA CUDA kernel; correct but no gain (-0.3%)",
    ("kda", 10): "packed KDA decode fast path kept for K3's lower-bounded gate + overlap (-4.6%, bit-exact; gate FAIL by 0.4pt)",
    ("kda", 11): "fast path + Triton recurrent-kernel launch tuning (num_warps) + overlap (-4.4%, bit-exact)",
    ("kda", 20): "infra: judge OOM (167 GB co-tenant on the shared GPU); not a result",
    ("kda", 21): "route+quant specialization broke numerics at B=32/1 (rel 0.42) -> correctness FAIL",
    ("kda", 22): "ACCEPTED: bf16-state port of the fused KDA decode JIT kernel (kda_fused_decode.cuh + .py): +1.3% @128, +2.5% @32, +3.1% @1",
    ("kda", 23): "null: five ideas rejected by the agent itself; +0.3% below the 0.5% floor",
    ("kda", 24): "null: tree returned to the seed",
    ("kda", 25): "null: +0.3% below the 0.56% needed (KDA plateau)",
    ("mla", 4): "campaign 1: cute-dsl -> trtllm-gen MLA decode + shared-expert overlap (-11.3% @1x1M, -2.1% @128x8k)",
    ("mla", 5): "campaign 1: cute-dsl -> trtllm-gen MLA decode (-14.5% @1x1M)",
    ("mla", 6): "campaign 1: see agent_iterations.md",
    ("mla", 7): "campaign 1 (re-run with the correct MLA oracle): see agent_iterations.md",
    ("mla", 8): "ACCEPTED: non-DCP decode routed cute-dsl -> TRT-LLM MLA generation kernel (15 lines, cutedsl_mla_backend.py): -12.1% @1x1M, bit-exact",
    ("mla", 20): "ACCEPTED: route_quant_fused JIT specialized for the 112-expert/top-2 shape (+0.75%; harness-specific, production is 896/top-16)",
    ("mla", 21): "ACCEPTED: fp32-output front GEMMs (15984x7168 / 6016x7168, m<=16) cuBLAS -> CuTe TGV (+1.5% @1x1M, +3.2% @128x8k)",
    ("mla", 22): "null: cutedsl_bf16_gemm tweak, 0.0%",
    ("mla", 23): "ACCEPTED: latent_up (7168x3584) and shared_down (7168x6144) -> BF16 TGV kernel (+1.5%)",
    ("mla", 24): "ACCEPTED: shared/routed alt-stream overlap in KimiK3MoE._forward_fused (+2.4%; +2.2% @128x8k, +3.4% @16x64k)",
    ("mla", 25): "ACCEPTED: is_var_seq=False (FlashInfer persistent TRT-LLM MLA schedule) for the fp8 K3 layout + 16-warp KV-concat CTA at B=1 (+0.8%); verified on a mixed-length batch (rel 0.0046, -7.4% there)",
    ("mla", 26): "null: cutedsl_bf16_gemm.py tweak, 0.0%",
    ("mla", 27): "null: fused finalize+shared JIT kernel regressed and was reverted; flashinfer_trtllm.py tweak 0.0% (MLA plateau)",
    ("kda_b512", 1): "Codex: null (+1.2% < 1.5% need under fill noise); MoE tuner ceiling, route+quant JITs inactive",
    ("kda_b512", 2): "Codex: null 0.0%; bf16-act MXFP4 (no kernel), route+pack+quant, TMA stages, tuner ceiling",
    ("kda_b512", 3): "Codex: null +0.1%; in-kernel TRT-LLM routing re-routed 2/512 tokens (CHECK FAIL), SiTU bf16 (no kernel)",
    ("mla_b512", 1): "INFRA NOISE (concurrent VibeSim profiling on the GPU); not a result",
    ("mla_b512", 2): "Codex: null 0.00%; tactic buckets, PDL, low-priority overlap (-1us), bf16 front GEMM (numerics FAIL)",
    ("mla_b512", 3): "Codex: null +0.17%; route+quant cap 64->512 (-1.5us kept), TGV at m=512 8%/2% SLOWER (reverted)",
    ("mla_b512", 4): "Codex: null 0.00%; PDL policy regressed, var-seq scheduler neutral",
    ("kda_b512_claude", 1): "ACCEPTED (Claude Opus 5.5): one-shot MXFP4 MoE autotune when rows/expert>8 (= production warmup, HARNESS GAP ~9 pts) + vectorized bf16 state ld/st & launch_bounds in kda_fused_decode.cuh (92.6->83.0us, real) + bfa overlap limit 128->512: 675.2->600.4 @512, exact",
    ("kda_b512_claude", 2): "ACCEPTED on the 15-rep re-judge (2026-09-27, follow-up #2): fused-kernel prologue load hoist + quad-row warp reduction (80.4->76.6us kernel): 599.5->593.5 @512 (+1.0%, sigma 0.3us), +1.1% @256, +0.5% @128, rel 0.017; the original 5-rep judge lost it to noise (sigma 7.3us, 3.65% need)",
    ("mla_b512_claude", 2): "ACCEPTED (Claude): tail split of the persistent MLA decode kernel's last wave (B=148*3+68 -> split 68 trailing requests into 136 half-KV pseudo-requests, LSE merge; mla_decode_tail_split.py): 904.6->888.0 @512, exact",
    ("kda_b512_claude", 3): "Claude: FAIL correctness (rel 0.029/0.022/0.021, 8/1/1 rows over tol) -- attn-res TMA fused residual add (fp32 add + RNE) changes rounding; the r2 kernel port (via the KB) and routed-before-shared capture order were exact (-3.1% total @512)",
    ("mla_prefill_claude", 3): "Claude: null (re-judged clean after a foreign-job-noise verdict): -0.5% @pf49152, +2.0% 4x4k; agent hit its cap without an accepted change",
    ("mla_prefill_claude", 2): "ACCEPTED (Claude): non-causal prefix attention TRT-LLM ragged FMHA -> sglang CuTe-DSL JIT FMHA with full softmax correction (fp8 P prescale 2^8): 11539->11229 @pf49152 (-2.7%), rel 0.0156",
    ("kda_prefill_claude", 3): "Claude: FAIL below 3-sigma (+0.95% first chunk vs 1.71% need; +2.2% @pf49152; correct); no agent log, tree carries the r2 h-scan kernel",
    ("kda_prefill_claude", 2): "Claude: FAIL below 3-sigma (+1.07% first chunk vs 1.33% need; +2.9% @pf49152, +1.7% 4x4k; bit-exact): CUDA kda_chunk_h h-scan kernel replacing Triton chunk_delta_h (516->281us) -- real, worth porting",
    ("kda_prefill_claude", 1): "ACCEPTED (Claude, re-judged after a GPU-2 OOM): KDA prefill copy removal (strided l2norm q/k, strided o_norm gate, strided v in recompute_w_u) + dropped int(query_start_loc[-1]) host sync; 9007->8584 self-measured, bit-exact; first judge OOM'd (tenant on GPU 2), re-judged",
    ("mla_prefill_claude", 1): "ACCEPTED (Claude): Triton mla_kv_pack_quantize_fp8 (one-pass K/V pack + fp8 quant for chunk+prefix, Q cast once; replaces ~800us elementwise) + residual add fused into attn-res TMA (418->361us @16k): 12128->11573 @pf49152 (-4.6%), -4.0% first chunk, -2.6% 4x4k; bit-exact",
    ("mla_b512_claude", 3): "ACCEPTED (Claude): output-gate GEMM launched into the MLA decode kernel's tail wave + pending residual add fused into the attn-res TMA kernel (addend in the TMA ring): 887.3->877.8 @512, mixed -3.0%, exact",
    ("mla_b512_claude", 1): "ACCEPTED (Claude Opus 5.5): tuned MoE tactic (HARNESS GAP ~6.4 pts) + fp8 set_mla_kv_concat_q satfinite cvt + exact NOSAT fixup (17.8->8.0us, real): 976.3->904.6 @512, exact; merged-front split rejected (slower)",
    # shape-matched campaign (2026-09-26/27, slurm mode, seed = b512 Claude best tree; same 5 shapes for both layers)
    ("kda_shapes_claude", 1): "ACCEPTED (Claude, slurm mode): split the merged MoE front at m<=32 (routed rows first, gate_up+shared on the alt stream, latent tail before the join) + fp32 TGV halves: 124.3->109.9 @1x1M (-11.5%), -13.0% @1x8k, -5.4% @32x8k, 128x8k flat; bit-exact at B=1/16",
    ("kda_shapes_claude", 2): "ACCEPTED (Claude): L2 prefetch JIT kernel (l2_prefetch.cuh) pulling o_proj + routed-front weights into L2 on the alt stream during the KDA recurrence (32 CTAs): 109.9->103.8 @1x1M (-5.6%), -3.5% @16x64k; 1x8k +0.9% (noise floor)",
    ("kda_shapes_claude", 3): "ACCEPTED (Claude): fused-decode / attn-res kernel tuning on top of r2: 103.8->101.8 @1x1M (-2.0%), -7.2% @1x8k, -2.4% @16x64k; cumulative vs pristine 136.6->101.8 (-25.5%)",
    ("mla_shapes_claude", 1): "ACCEPTED (Claude, slurm mode): B=1 KV split k=8 + LSE merge for L>=64k (256MB MLA workspace) and set_mla_kv_concat_q_fp8 writing the split q replicas: 241.1->216.6 @1x1M (-10.2%), other 4 shapes flat, rel 0.006",
    ("mla_shapes_claude", 2): "ACCEPTED (Claude): attn-res cluster_small kernel + fused MoE finalize+RMSNorm: 218.4->216.4 @1x1M (-0.9%, above the 3-sigma floor), -1.6% @1x8k; near the 1x1M plateau",
    ("mla_shapes_claude", 3): "null (Claude): custom small-m MXFP4 MoE kernel (mxfp4_small_m_moe.cuh) -- correct but +0.4% @1x1M; MLA 1x1M plateau at ~216us",
    # long-context chunked prefill (2026-09-27, slurm mode; 16k/32k chunks at 128k/262k contexts; seed = prefill Claude best tree)
    ("kda_lcprefill_claude", 1): "ACCEPTED (Claude): CUDA mma.sync h-scan kernel (kda_chunk_h.cuh, ported from the earlier unaccepted prefill r2 via the KB) + conv1d BLOCK_M=16/nw=2: 8612.9->8471.6 us @16k/pf245760 (-1.6%), -3.1% @16k/pf131072, bit-exact",
    ("kda_lcprefill_claude", 2): "ACCEPTED (Claude): fused SiLU-and-mul JIT kernel for the MoE activation (situ_and_mul.cuh): 8307.9->8193.2 @16k/pf245760 (-1.4%), -4.0% @16k/pf131072, rel 0.007; side-stream in-proj overlap rejected by its own A/B (<=0.3%: persistent GEMMs serialize)",
    ("kda_lcprefill_claude", 3): "FAIL below 3-sigma (Claude): +0.7% @16k/pf245760 (secondaries -1.6..-3.6%), correct; no iteration log. Re-judged at 15 reps (2026-09-27): +0.9% vs a 1.31% floor (prefill sigma ~0.4% is intrinsic) -> still not accepted; a real but sub-1% effect",
    ("mla_lcprefill_claude", 1): "ACCEPTED (Claude): 16B-align the prefix-chunk cum_seqlen_k row so the accepted CuTe-DSL prefix FMHA actually engages at long prefixes (it silently fell back to trtllm-gen; prefix FMHA = 66% of the step) + empirical KV-split policy (S=4 <=6 waves else S=2): 25196.8->21998.0 @16k/pf245760 (-12.7%), -7.5/-7.4/-4.4% on the others, bit-exact",
    ("mla_lcprefill_claude", 2): "ACCEPTED (Claude): prefix-FMHA softmax: 40 of 128 exp2 per tile emulated on the FMA pipe with a degree-2 polynomial (MUFU relief): 22001.9->20967.8 @16k/pf245760 (-4.7%), -5.7/-2.1/-3.1%, bit-exact",
    ("mla_lcprefill_claude", 3): "FAIL correctness (Claude): +1.4% @16k/pf245760 but rel 0.30 / row-wise rule failed at 32k/pf229376 (rel 0.25 on the primary, passed by rows) -- a numerics-changing prefix-attention change; rejected",
    # follow-ups 2026-09-27 (rounds 4-6): B12c-corrected long-context prefill oracles (8807/8808) from MLA r6 / KDA r4 on
    ("mla_lcprefill_claude", 4): "null (Claude): -0.4% @16k/pf245760, correct; agent hit its budget on one hypothesis",
    ("mla_lcprefill_claude", 5): "ACCEPTED (Claude): fp8 packing of the prefix-chunk kv_b_proj output fused into the GEMM epilogue (mla_kv_b_proj_pack_fp8 + dense_gemm_sm100_fp8_via_bf16_epilogue): 20837.8->20609.2 @16k/pf245760 (-1.1%), -0.7% @16k/pf131072, bit-exact; a single-launch prefix FMHA was tried and reverted (neutral)",
    ("mla_lcprefill_claude", 6): "ACCEPTED (Claude, corrected oracle 8808): causal in-chunk attention pass on the attention alt stream overlapping the prefix passes, joined before merge_state: 20791.6->20452.7 @16k/pf245760 (-1.6%), -0.8% @16k/pf131072, bit-exact; cumulative 27.69->20.45 ms (-26.1%)",
    ("kda_lcprefill_claude", 4): "ACCEPTED (Claude, 60-min budget, corrected oracle 8807): warp-specialized 4-stage kda_chunk_h (from the rejected r3 tree) + pinned BK32/nw1 config for the inter-chunk solve: 8264.9->8133.9 @16k/pf245760 (-1.6%), -3.5% @16k/pf131072, rel 0.007; cumulative 9.03->8.13 ms (-10.0%)",
    ("kda_lcprefill_claude", 5): "FAIL below 3-sigma (Claude): -1.2% @16k/pf245760, -3.2% @16k/pf131072, correct (rel 0.017); 15-rep re-judge: -1.1% vs a 3.09% floor (sigma 85 us, noisy window) -> still unaccepted; a real ~1-3% effect",
    ("kda_lcprefill_claude", 6): "FAIL below 3-sigma (Claude): fused conv+l2norm and chunk_intra CUDA kernels (kda_conv_l2norm.cuh, kda_chunk_intra.cuh): -1.2% @16k/pf245760, -3.0% @16k/pf131072, -2.5% @32k/pf131072; passes the row-wise rule (rows over tol within budget) but max_rel 0.30 -> a numerics-changing rewrite; not accepted",
    # follow-up #4 (2026-09-27/28, slurm mode): speculative VERIFY (TARGET_VERIFY, vk3 = 4 tokens/request) and MIXED
    # prefill-chunk + decode batches. Verify seeds: KDA <- shapes best tree, MLA <- PRISTINE (every accepted MLA tree
    # regressed the verify step). Mixed seeds: the layer's lcprefill best tree. Verify metric switched from the eager
    # step (65% host launch time, sigma ~90-400 us) to CUDA-graph replay (--verify-graph) after KDA r1/r2.
    ("kda_verify_claude", 1): "FAIL by noise (Claude, EAGER metric): 13 files (fused verify recurrence variants, conv, norm-gate, MoE gate) -21% @64x8k vk3 at 5 reps / -14.7% at 15 reps, correct -- but the eager verify step's sigma (92 us, 7%) put the 3-sigma floor at 22%. Graph-mode re-judge: -0.05% -> the whole eager gain was host launch time, not GPU work.",
    ("kda_verify_claude", 2): "FAIL by noise (Claude, EAGER metric): -21.7% eager, correct, sigma 418 us (99% floor). Graph-mode re-judge (rejudge_vkgraph_trial_2): PASS -0.62% @64x8k, -1.1% @16x64k -- real but small; superseded by r3 (separate branch), not promoted.",
    ("kda_verify_claude", 3): "ACCEPTED (Claude, GRAPH metric): CUDA JIT verify recurrence kernel (kda_verify_recurrent.cuh, replaces the Triton fused_recurrent verify path; 2x off roofline before): 490.0->472.5 @64x8k vk3 (-3.55%), -4.85% @128x8k, -2.1% @16x64k, rel 0.010-0.012. Neutral/worse: token-major conv1d out=, store-policy/occupancy/TMA variants of the verify kernel.",
    ("mla_verify_claude", 1): "ACCEPTED (Claude, GRAPH metric, from pristine): satfinite fp8 cvt in set_mla_kv_concat_q + shared/routed alt-stream overlap + MLA output-gate g_proj enqueued after the MLA node (gate fork event at forward_absorb_core, overlap limit 512 tokens): 492.9->472.4 @64x8k vk3 (-4.15%), -5.0% @128x8k, -5.8% @16x64k, bit-exact. Ruled out by microbench: trtllm-gen MLA at q_len=4 (2x slower), split_kv override, TGV/deep_gemm GEMM swaps.",
    ("mla_verify_claude", 2): "null (Claude): variants of the same KV-concat kernel, -0.18%, exact.",
    ("mla_verify_claude", 3): "ACCEPTED (Claude, GRAPH metric): residual add fused into the attn_res TMA aggregate + SM carveout on the shared-expert down GEMM so the MoE routing kernel is not blocked: 483.7->473.6 @64x8k vk3 (-2.09%), -0.3%/-0.8% secondaries, bit-exact. Ruled out: forking the shared experts after topk (482.5).",
    ("kda_mixed_claude", 1): "FAIL (Claude): edited blind while every GPU request sat behind slurm-manager placeholders for 5 h; tree +52..77% slower (first verdict 'non-compiling' was the smoke timing out in the queue).",
    ("kda_mixed_claude", 2): "ACCEPTED (Claude, eager metric): lift the one-wave dispatch guard (cdiv(V,16)*N*H <= #SMs, measured on the old single-stage kernel) so the 4-stage cp.async CUDA kda_chunk_h scan also serves mixed batches (N=65/129 sequences; Triton h-scan was 645 us there): 8659->8361 us @64 dec + 16k chunk (-3.4%), -4.5% @48k prefix, 128-dec point flat; row-wise CHECK pass.",
    ("mla_mixed_claude", 1): "VOID (Claude): tree byte-identical to the seed -- the agent never got past iter_00 while GPU-starved (placeholders); +0.04%.",
}
CAMPAIGN1 = {("kda", k) for k in (4, 5, 6, 7)} | {("mla", k) for k in (4, 5, 6, 7)}
# The tree a case's round 1 started from when it is another case's best tree (else stacked_tree_judged / pristine).
SEEDS = {
    "kda_b512_claude": "kda/best_tree", "mla_b512_claude": "mla/best_tree",
    "kda_prefill_claude": "kda/best_tree", "mla_prefill_claude": "mla/best_tree",
    "kda_shapes_claude": "kda_b512_claude/best_tree", "mla_shapes_claude": "mla_b512_claude/best_tree",
    "kda_lcprefill_claude": "kda_prefill_claude/best_tree", "mla_lcprefill_claude": "mla_prefill_claude/best_tree",
    "kda_mixed_claude": "kda_lcprefill_claude/best_tree", "mla_mixed_claude": "mla_lcprefill_claude/best_tree",
    "kda_verify_claude": "kda_shapes_claude/best_tree",
}


def sh(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw).stdout


def trial_k(path):
    m = re.match(r"trial_(\d+)_verdict\.json$", path.name)
    return int(m.group(1)) if m else None


def src_only(diff_text):
    """Keep only hunks of source files (drop pycache/binary noise)."""
    out, keep = [], False
    for line in diff_text.splitlines():
        if line.startswith("diff ") or line.startswith("Only in "):
            keep = line.endswith(SRC_EXT) or any(line.rstrip().endswith(e) for e in SRC_EXT)
            if line.startswith("Only in ") and "__pycache__" in line:
                keep = False
        if keep:
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def agent_iterations(opt_run):
    parts = []
    log_md = opt_run / "log.md"
    if log_md.exists():
        parts.append("## agent log.md\n\n" + log_md.read_text(errors="replace").strip() + "\n")
    for it in sorted(opt_run.glob("iter_*")):
        sec = [f"## {it.name}"]
        for name in ("hypothesis.md", "analysis.md"):
            f = it / name
            if f.exists():
                txt = f.read_text(errors="replace").strip()
                if len(txt) > 6000:
                    txt = txt[:6000] + "\n[... truncated]"
                sec.append(f"### {name}\n\n{txt}\n")
        rj = it / "result.json"
        if rj.exists():
            txt = rj.read_text(errors="replace").strip()
            sec.append("### result.json\n\n```\n" + (txt[:2500] + ("\n[... truncated]" if len(txt) > 2500 else "")) + "\n```\n")
        dp = it / "diff.patch"
        if dp.exists():
            try:
                n = sum(1 for _ in open(dp, errors="replace"))
            except OSError:
                n = -1
            sec.append(f"(diff.patch: {n} lines, files: "
                       + ", ".join(sorted(set(re.findall(r"^\+\+\+ .*?(python/sglang/\S+|\S+\.(?:py|cuh?|h))", dp.read_text(errors='replace'), re.M))))[:400] + ")\n")
        if len(sec) > 1:
            parts.append("\n".join(sec))
    return "\n".join(parts)


def build_case(case, out, hist):
    d = HERE / f"iter_opt_eval_k3_{case}"
    if not d.is_dir():
        return
    rounds = {}
    rj = d / "rounds.jsonl"
    if rj.exists():
        for line in rj.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rounds[r["round"]] = r
    verdicts = sorted([(trial_k(p), p) for p in d.glob("trial_*_verdict.json") if trial_k(p) is not None])
    # continuous rounds: the tree a round started from = judged tree of the previous PASS round,
    # else the seed (kda: stacked_tree_judged; mla: pristine).
    prev_pass_tree = None
    seed_tree = d / "stacked_tree_judged" if (d / "stacked_tree_judged").is_dir() else d / "pristine_tree"
    if case in SEEDS and (HERE / f"iter_opt_eval_k3_{SEEDS[case]}").is_dir():
        seed_tree = HERE / f"iter_opt_eval_k3_{SEEDS[case]}"
    for k, vp in verdicts:
        v = json.loads(vp.read_text())
        tdir = out / "trials" / f"{case}_{k}"
        tdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(vp, tdir / "verdict.json")
        patch = d / f"trial_{k}_tree_judged.patch"
        if patch.exists():
            shutil.copy(patch, tdir / "patch.diff")
        continuous = bool(v.get("baseline_tree")) or k in rounds
        incr_lines = None
        if continuous and (d / f"trial_{k}_tree_judged").is_dir():
            base = prev_pass_tree or seed_tree
            if base.is_dir():
                diff = sh(["diff", "-ruN", "-x", "__pycache__", str(base), str(d / f"trial_{k}_tree_judged")])
                diff = src_only(diff)
                (tdir / "incremental.diff").write_text(diff)
                incr_lines = diff.count("\n")
        opt_run = d / f"trial_{k}_opt_run"
        if opt_run.is_dir():
            txt = agent_iterations(opt_run)
            if txt.strip():
                (tdir / "agent_iterations.md").write_text(f"# {case.upper()} trial {k}: agent iterations\n\n" + txt)
        pts = {}
        for pk, p in (v.get("points") or {}).items():
            pts[pk] = {"before_us": round(p["before_us"], 1), "after_us": round(p["after_us"], 1),
                       "improvement": round(p["improvement"], 4),
                       "check": p["correctness"].get("pass"), "rel_err": p["correctness"].get("max_rel_err")}
        rec = {"case": case, "trial": k, "verdict": v.get("verdict"), "reason": v.get("reason"),
               "continuous_round": continuous, "campaign1_unrealistic_routing": (case, k) in CAMPAIGN1,
               "points": pts, "files": v.get("changed_files") or [], "diff_lines": v.get("diff_lines"),
               "incremental_diff_lines": incr_lines, "note": NOTES.get((case, k), ""),
               "dir": f"trials/{case}_{k}"}
        hist.append(rec)
        if continuous and v.get("verdict") == "PASS" and (d / f"trial_{k}_tree_judged").is_dir():
            prev_pass_tree = d / f"trial_{k}_tree_judged"
    # extra judged trees (stacked, seed rechecks, mixed point) -> plain copies of verdicts
    for extra in ("stacked_verdict.json", "best_tree_seed1_verdict.json", "best_tree_seed2_verdict.json",
                  "best_tree_mixed_verdict.json", "rejudge15_trial_1_verdict.json", "rejudge15_trial_2_verdict.json",
                  "rejudge15_trial_3_verdict.json", "rejudge_vkgraph_trial_1_verdict.json", "rejudge_vkgraph_trial_2_verdict.json"):
        if (d / extra).exists():
            (out / "trials" / f"{case}_extra").mkdir(parents=True, exist_ok=True)
            shutil.copy(d / extra, out / "trials" / f"{case}_extra" / extra)


def write_index(out, hist):
    lines = ["# Kimi-K3 optimization history (warm start index)", "",
             f"*generated {time.strftime('%F %T')} by build_opt_history.py; read TECHNIQUES.md first*", "",
             "Each row is one judged trial/round. `Δ` = latency change per point (positive = faster), "
             "`chk` = output+state match vs the pristine goldens. Continuous rounds start from the previous "
             "accepted tree; their `incremental.diff` is what that round changed. Campaign-1 rows (c1) ran "
             "on a workload with collapsed MoE routing -- directions are valid, effect sizes are not.", "",
             "| case | trial | verdict | points (before→after µs, Δ, chk) | files | note | dir |",
             "|---|---|---|---|---|---|---|"]
    for r in hist:
        pts = "; ".join(f"{k}: {p['before_us']}→{p['after_us']} ({p['improvement']*100:+.1f}%, {'ok' if p['check'] else 'FAIL'})"
                        for k, p in r["points"].items())
        files = ", ".join(pathlib.PurePosixPath(f).name for f in r["files"])[:160]
        tag = " (c1)" if r["campaign1_unrealistic_routing"] else ""
        lines.append(f"| {r['case']} | {r['trial']}{tag} | {r['verdict']} | {pts} | {files} | {r['note']} | `{r['dir']}` |")
    lines += ["", "## How to use", "",
              "1. `TECHNIQUES.md`: accepted levers (already in your tree if you were seeded with a best tree) and dead ends with the reason each failed.",
              "2. `grep -ril <kernel or file name> trials/*/agent_iterations.md` to see every hypothesis ever tried around that code, with the measured result.",
              "3. `trials/<case>_<k>/incremental.diff` is the minimal patch of an accepted round -- the fastest way to transfer a lever to a new workload point.",
              "4. Workloads differ (decode B=128 vs B=512 vs chunked prefill): a dead end at one point can be a win at another and vice versa. Re-measure; never assume.",
              ""]
    (out / "INDEX.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "k3_opt_history"))
    ap.add_argument("--cases", default="kda,mla,kda_b512,mla_b512,kda_b512_claude,mla_b512_claude,kda_prefill,mla_prefill,kda_prefill_claude,mla_prefill_claude,"
                                       "kda_shapes_claude,mla_shapes_claude,kda_lcprefill_claude,mla_lcprefill_claude,"
                                       "kda_verify_claude,mla_verify_claude,kda_mixed_claude,mla_mixed_claude")
    a = ap.parse_args()
    final = pathlib.Path(a.out)
    # build into a sibling temp dir and rsync into place: the final dir is bind-mounted read-only into
    # running agent containers, so it must never disappear while a trial is in flight.
    out = final.parent / (final.name + ".build"); shutil.rmtree(out, ignore_errors=True)
    (out / "trials").mkdir(parents=True)
    hist = []
    for case in a.cases.split(","):
        build_case(case.strip(), out, hist)
    hist.sort(key=lambda r: (r["case"], r["trial"]))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    tech = HERE / "opt_history_techniques.md"
    if tech.exists():
        shutil.copy(tech, out / "TECHNIQUES.md")
    write_index(out, hist)
    final.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", str(out) + "/", str(final) + "/"], check=True)
    shutil.rmtree(out); out = final
    n_pass = sum(1 for r in hist if r["verdict"] == "PASS")
    print(f"wrote {out}: {len(hist)} trials ({n_pass} PASS), "
          f"{sum(1 for p in out.rglob('agent_iterations.md'))} agent_iterations.md, "
          f"{sum(1 for p in out.rglob('incremental.diff'))} incremental diffs")


if __name__ == "__main__":
    main()
