#!/usr/bin/env python3
"""Trajectory audit: did the agent use the VibeSim oracle before each accepted change?
For each (case, trial): count oracle API calls per verb in the agent transcript (Codex plain log or Claude
stream-json), list the VibeSim nodes named in the agent's analysis.md / log.md, and print the sentences that
mention VibeSim. Output: markdown to stdout."""
import json, pathlib, re, sys
HERE = pathlib.Path(__file__).resolve().parent
VERBS = ["workspace-info", "simulate", "analyze", "optimality", "kernels"]
NODES = ["mxfp4_fused_moe", "mla_decode_attention", "merged_front", "shared_down", "latent_up", "kda_recurrent_decode",
         "kda_fused_decode", "qkvbfg", "attention subtree", "front GEMM", "mla_prefill_attention", "kda_chunk_prefill",
         "mla_cache_append", "o_proj", "routed", "K/V pack", "attn_res", "elementwise"]

def oracle_calls(log):
    counts = {v: 0 for v in VERBS}
    if not log.exists(): return counts, 0
    txt = log.read_text(errors="replace")
    # Claude stream-json: count tool_use commands; Codex: count 'exec' command lines. Fallback: any line with api/v1.
    n_lines = 0
    for line in txt.splitlines():
        if "api/v1/" not in line: continue
        if line.startswith("{"):  # stream-json event
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "assistant": continue
            for c in d.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    cmd = json.dumps(c.get("input"))
                    for v in VERBS:
                        counts[v] += len(re.findall(r"api/v1/" + re.escape(v), cmd))
                    n_lines += 1
        else:
            if "curl" in line or "requests.get" in line or "urlopen" in line or "http.get" in line:
                for v in VERBS:
                    counts[v] += len(re.findall(r"api/v1/" + re.escape(v), line))
                n_lines += 1
    return counts, n_lines

def vibesim_sentences(opt_run, limit=4):
    out = []
    for f in sorted(opt_run.glob("**/*.md")):
        for s in re.split(r"(?<=[.;])\s+|\n", f.read_text(errors="replace")):
            if re.search(r"VibeSim|optimality|necessary_share|R0/R5|cached alternative|has_cached_alternative", s):
                s = " ".join(s.split())
                if 40 < len(s) < 400: out.append((f.name, s))
    seen, res = set(), []
    for f, s in out:
        k = s[:80]
        if k in seen: continue
        seen.add(k); res.append((f, s))
        if len(res) >= limit: break
    return res

def nodes_named(opt_run):
    txt = " ".join(f.read_text(errors="replace") for f in opt_run.glob("**/analysis.md")) if opt_run.exists() else ""
    txt += " ".join(f.read_text(errors="replace") for f in opt_run.glob("log.md"))
    return [n for n in NODES if n in txt]

def audit(case, k):
    d = HERE / f"iter_opt_eval_k3_{case}"
    log = d / f"trial_{k}_agent.log"; opt = d / f"trial_{k}_opt_run"
    counts, n = oracle_calls(log)
    return {"case": case, "trial": k, "calls": counts, "total": sum(counts.values()), "nodes": nodes_named(opt),
            "quotes": vibesim_sentences(opt), "log": str(log), "opt_run": str(opt)}

if __name__ == "__main__":
    spec = [("mla", 8), ("mla", 20), ("mla", 21), ("mla", 23), ("mla", 24), ("mla", 25),
            ("kda", 8), ("kda", 10), ("kda", 11), ("kda", 22),
            ("mla_b512_claude", 1), ("mla_b512_claude", 2), ("mla_b512_claude", 3), ("kda_b512_claude", 1),
            ("mla_prefill_claude", 1), ("mla_prefill_claude", 2), ("kda_prefill_claude", 1)]
    if len(sys.argv) > 1: spec = [tuple(a.rsplit(":", 1)) for a in sys.argv[1:]]; spec = [(c, int(k)) for c, k in spec]
    for c, k in spec:
        r = audit(c, k)
        print(f"## {c} trial {k}: oracle calls {r['total']} " + " ".join(f"{v}={n}" for v, n in r["calls"].items()))
        print("nodes named in analysis:", ", ".join(r["nodes"]) or "(none)")
        for f, s in r["quotes"]: print(f"  [{f}] {s}")
        print()
