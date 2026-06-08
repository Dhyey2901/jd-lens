"""Eval harness — validate scorer + signal outputs against ground-truth pairs.

Usage:
    python eval/run_eval.py              # run all cases
    python eval/run_eval.py --fast       # skip embedding model, signals-only
    make eval                            # same as above via Makefile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor import extract_jd
from signals import compute_hiring_signals, generate_prediction


def _check(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def run_signals_only(cases: list[dict]) -> list[dict]:
    """Evaluate only rule-based signal scoring — no embedding model needed."""
    results = []
    for case in cases:
        jd_info = extract_jd(case["jd"])
        hiring = compute_hiring_signals(
            case["resume"],
            jd_soft_skills=jd_info.get("soft_skills", []),
            jd_text=case["jd"],
            jd_skills=jd_info.get("skills_and_tools", []),
            missing_skills=[],  # no scorer output — conservative (no penalty)
            skill_weights=jd_info.get("skill_weights", {}),
        )
        exp = case["expected"]
        signal_ok = _check(hiring["hiring_signal_score"], *exp["signal_range"])
        results.append({
            "id": case["id"],
            "description": case["description"],
            "signal": hiring["hiring_signal_score"],
            "signal_range": "{}-{}".format(*exp["signal_range"]),
            "signal_ok": signal_ok,
            "role_type": hiring["role_type"],
            "mode": "signals-only",
        })
    return results


def run_full(cases: list[dict]) -> list[dict]:
    """Full evaluation including embedding model for JD match score."""
    from scorer import compute_fit_score, get_embedding_model

    print("Loading embedding model (all-MiniLM-L6-v2)…")
    model = get_embedding_model()

    results = []
    for case in cases:
        t0 = time.time()
        jd_info = extract_jd(case["jd"])
        score_info = compute_fit_score(
            case["jd"], case["resume"],
            jd_skills=jd_info["skills_and_tools"],
            embedding_model=model,
        )
        hiring = compute_hiring_signals(
            case["resume"],
            jd_soft_skills=jd_info.get("soft_skills", []),
            jd_text=case["jd"],
            jd_skills=jd_info.get("skills_and_tools", []),
            missing_skills=score_info["missing_skills"],
            skill_weights=jd_info.get("skill_weights", {}),
        )
        prediction = generate_prediction(
            jd_match=score_info["fit_score"],
            signal_score=hiring["hiring_signal_score"],
            missing_skills=score_info["missing_skills"],
        )
        elapsed = time.time() - t0

        exp = case["expected"]
        verdict_ok  = prediction["verdict"] == exp["verdict"]
        jd_range_ok = _check(score_info["fit_score"], *exp["jd_match_range"])
        sig_range_ok = _check(hiring["hiring_signal_score"], *exp["signal_range"])

        results.append({
            "id":           case["id"],
            "description":  case["description"],
            "verdict":      prediction["verdict"],
            "exp_verdict":  exp["verdict"],
            "verdict_ok":   verdict_ok,
            "jd_match":     score_info["fit_score"],
            "jd_range":     "{}-{}".format(*exp["jd_match_range"]),
            "jd_range_ok":  jd_range_ok,
            "signal":       hiring["hiring_signal_score"],
            "signal_range": "{}-{}".format(*exp["signal_range"]),
            "sig_range_ok": sig_range_ok,
            "role_type":    hiring["role_type"],
            "elapsed_s":    round(elapsed, 2),
            "mode":         "full",
        })

    return results


def print_report(results: list[dict]) -> int:
    mode = results[0]["mode"] if results else "full"
    full = mode == "full"

    if full:
        passes = sum(
            1 for r in results
            if r["verdict_ok"] and r["jd_range_ok"] and r["sig_range_ok"]
        )
    else:
        passes = sum(1 for r in results if r["signal_ok"])

    total = len(results)

    print(f"\n{'='*72}")
    print(f"JD LENS EVAL — {passes}/{total} passed  ({mode} mode)")
    print(f"{'='*72}")

    for r in results:
        if full:
            ok = r["verdict_ok"] and r["jd_range_ok"] and r["sig_range_ok"]
        else:
            ok = r["signal_ok"]

        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\n{status}  [{r['id']}]")
        print(f"       {r['description']}")

        if full:
            v_mark = "✓" if r["verdict_ok"]  else "✗"
            j_mark = "✓" if r["jd_range_ok"] else "✗"
            s_mark = "✓" if r["sig_range_ok"] else "✗"
            print(f"  Verdict:  {r['verdict']:<35} expected: {r['exp_verdict']}  {v_mark}")
            print(f"  JD Match: {r['jd_match']:>5.1f}%  (expected {r['jd_range']}%)  {j_mark}")
            print(f"  Signal:   {r['signal']:>5.1f}%  (expected {r['signal_range']}%)  {s_mark}")
            print(f"  Role type: {r['role_type']}   |  {r['elapsed_s']}s")
        else:
            s_mark = "✓" if r["signal_ok"] else "✗"
            print(f"  Signal:   {r['signal']:>5.1f}%  (expected {r['signal_range']}%)  {s_mark}")
            print(f"  Role type: {r['role_type']}")

    print(f"\n{'='*72}")
    accuracy = 100 * passes // total if total else 0
    print(f"Accuracy: {passes}/{total} = {accuracy}%")
    if passes < total:
        print("Some cases failed — review expected ranges in eval/ground_truth.json")
    return 0 if passes == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="JD Lens eval harness")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip embedding model; validate signal scores only",
    )
    args = parser.parse_args()

    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path) as f:
        cases = json.load(f)

    results = run_signals_only(cases) if args.fast else run_full(cases)
    sys.exit(print_report(results))


if __name__ == "__main__":
    main()
