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
}
CAMPAIGN1 = {("kda", k) for k in (4, 5, 6, 7)} | {("mla", k) for k in (4, 5, 6, 7)}


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
                  "best_tree_mixed_verdict.json", "rejudge15_trial_2_verdict.json", "rejudge15_trial_3_verdict.json"):
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
                                       "kda_shapes_claude,mla_shapes_claude,kda_lcprefill_claude,mla_lcprefill_claude")
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
