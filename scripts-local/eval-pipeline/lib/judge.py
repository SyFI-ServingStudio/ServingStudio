#!/usr/bin/env python3
"""Semantic judge: does the agent's diagnosis match the ground-truth root cause?

Runs a gpt-5.6-luna judge conversation (single mode) and asks for a strict JSON
verdict. Independent of the objective sim gate.
"""
from __future__ import annotations
import json, re, sys
import vibesim_client as vc

JUDGE_RUNTIME = {"model": "gpt-5.6-luna", "effort": "high", "serviceTier": "default"}

PROMPT = """You are an impartial evaluator. An AI agent was asked to find and remove the
largest source of redundant work in a simulated ML model, WITHOUT being told the
answer. Below is (A) the ground-truth root cause of the real optimization, and
(B) the agent's own report. Decide whether the agent independently identified the
SAME root cause and proposed a change with the same effect.

=== (A) GROUND-TRUTH ROOT CAUSE ===
{gt}

=== (B) AGENT REPORT ===
{agent}

Respond with ONLY a JSON object, no prose, of exactly this shape:
{{"matches_root_cause": true|false,
  "identified_operator": "<the operator/sub-op the agent named, or 'none'>",
  "same_effect_fix": true|false,
  "score": <float 0.0-1.0>,
  "reasoning": "<=2 sentences"}}
matches_root_cause = the agent pinpointed the same bottleneck/root cause that
(A) describes. same_effect_fix = the change the agent actually IMPLEMENTED would
have the same effect as the ground-truth fix while preserving the necessary
computation. Judge the implemented fix and its diagnosis; if the report surveys
several candidate operators, weigh the one the agent chose to fix. Be strict:
fixing a different operator than (A)'s, or only switching a config/arch selector
without addressing the root cause, is NOT a match."""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"matches_root_cause": None, "score": None, "reasoning": "no JSON in verdict",
                "raw": text[:800]}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"matches_root_cause": None, "score": None, "reasoning": "unparseable JSON",
                "raw": m.group(0)[:800]}


def judge(wid: str, gt_root_cause: str, agent_report: str) -> dict:
    cid = vc.create_conversation(wid, runtime=JUDGE_RUNTIME, agent_mode="single", autonomous=True)
    res = vc.send_message(wid, cid, PROMPT.format(gt=gt_root_cause, agent=agent_report), timeout=1800)
    verdict = _extract_json(res.get("final", "") or "")
    verdict["_conversation_id"] = cid
    return verdict


if __name__ == "__main__":
    wid, gt_file, agent_file = sys.argv[1:4]
    print(json.dumps(judge(wid, open(gt_file).read(), open(agent_file).read()), indent=2))
