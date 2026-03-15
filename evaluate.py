"""
Evaluate metric agreement with human preference on VAQ-Bench.

For each pairwise instance in VAQ-Bench, the metric assigns scores to both
the candidate answer and the ground-truth answer.  Agreement is measured by
checking whether the metric's ranking direction matches the human label:
  - label  1 → candidate is better  → metric should score candidate higher
  - label -1 → candidate is worse   → metric should score candidate lower

Usage
-----
    python evaluate.py \
        --prediction-file results/my_metric.json \
        --benchmark-file  data/VAQ_Bench.json

Prediction file format (JSON):
    A dictionary keyed by question_id.  Each value is a dict mapping candidate
    type to a numeric score, e.g.:

    {
        "005-1": {
            "gt_score": 0.82,
            "Model_gpt5": 0.75,
            "Model_gemini2_5_pro": 0.68,
            "Noise_Hallucination": 0.55,
            "Noise_Mismatch": 0.42
        },
        ...
    }

    Alternatively, you may provide ``candidate_score`` and ``gt_score`` per
    candidate type:

    {
        "005-1": {
            "Model_gpt5": {"candidate_score": 0.75, "gt_score": 0.82},
            ...
        },
        ...
    }
"""

import argparse
import json
from collections import defaultdict


# ---------------------------------------------------------------------------
# Candidate type → setting mapping
# ---------------------------------------------------------------------------
MODEL_GENERATED = {"Model_gpt5", "Model_gemini2_5_pro", "Model_gemini2_5_flash"}
PERTURBATION = {"Noise_Hallucination", "Noise_Mismatch"}
ALL_TYPES = MODEL_GENERATED | PERTURBATION

SETTING_DISPLAY = {
    "Model_gpt5": "GPT-5",
    "Model_gemini2_5_pro": "Gemini-2.5-Pro",
    "Model_gemini2_5_flash": "Gemini-2.5-Flash",
    "Noise_Hallucination": "Hallucination",
    "Noise_Mismatch": "Mismatched",
}


def load_benchmark(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_agreement(benchmark: list[dict], predictions: dict) -> dict:
    """Return per-type and overall agreement rates."""
    correct = defaultdict(int)
    total = defaultdict(int)
    missing = 0

    for entry in benchmark:
        qid = entry["question_id"]
        if qid not in predictions:
            missing += len(entry.get("candidates", {}))
            continue

        pred_entry = predictions[qid]

        for ctype, cand_info in entry["candidates"].items():
            label = cand_info["label"]  # 1 or -1

            # ---- resolve scores ------------------------------------------------
            # Format 1: flat  {"gt_score": ..., "Model_gpt5": score, ...}
            # Format 2: nested {"Model_gpt5": {"candidate_score": ..., "gt_score": ...}}
            if isinstance(pred_entry.get(ctype), dict):
                cand_score = pred_entry[ctype].get("candidate_score")
                gt_score = pred_entry[ctype].get("gt_score")
            else:
                cand_score = pred_entry.get(ctype)
                gt_score = pred_entry.get("gt_score")

            if cand_score is None or gt_score is None:
                missing += 1
                continue

            # ---- agreement check -----------------------------------------------
            if label == 1 and cand_score > gt_score:
                correct[ctype] += 1
            elif label == -1 and cand_score < gt_score:
                correct[ctype] += 1
            # tie counts as incorrect (no direction matched)

            total[ctype] += 1

    results = {}
    for ctype in sorted(total.keys()):
        acc = correct[ctype] / total[ctype] * 100 if total[ctype] > 0 else 0.0
        results[ctype] = {
            "agreement": round(acc, 1),
            "correct": correct[ctype],
            "total": total[ctype],
        }

    # Aggregate: model-generated, perturbation, overall
    for group_name, group_types in [
        ("Model-Generated (avg)", MODEL_GENERATED),
        ("Controlled Perturbation (avg)", PERTURBATION),
        ("Overall", ALL_TYPES),
    ]:
        c = sum(correct[t] for t in group_types if t in total)
        n = sum(total[t] for t in group_types if t in total)
        results[group_name] = {
            "agreement": round(c / n * 100, 1) if n > 0 else 0.0,
            "correct": c,
            "total": n,
        }

    if missing:
        results["_missing"] = missing

    return results


def print_results(results: dict, metric_name: str = "Metric") -> None:
    print(f"\n{'=' * 62}")
    print(f"  VAQ-Bench Evaluation Results — {metric_name}")
    print(f"{'=' * 62}")

    # Model-generated
    print(f"\n  {'Model-Generated Answers':}")
    print(f"  {'─' * 50}")
    for ctype in ["Model_gpt5", "Model_gemini2_5_pro", "Model_gemini2_5_flash"]:
        if ctype in results:
            r = results[ctype]
            display = SETTING_DISPLAY.get(ctype, ctype)
            print(f"    {display:<25s} {r['agreement']:>5.1f}%  ({r['correct']}/{r['total']})")
    if "Model-Generated (avg)" in results:
        r = results["Model-Generated (avg)"]
        print(f"    {'Average':<25s} {r['agreement']:>5.1f}%  ({r['correct']}/{r['total']})")

    # Perturbation
    print(f"\n  {'Controlled Perturbations':}")
    print(f"  {'─' * 50}")
    for ctype in ["Noise_Hallucination", "Noise_Mismatch"]:
        if ctype in results:
            r = results[ctype]
            display = SETTING_DISPLAY.get(ctype, ctype)
            print(f"    {display:<25s} {r['agreement']:>5.1f}%  ({r['correct']}/{r['total']})")
    if "Controlled Perturbation (avg)" in results:
        r = results["Controlled Perturbation (avg)"]
        print(f"    {'Average':<25s} {r['agreement']:>5.1f}%  ({r['correct']}/{r['total']})")

    # Overall
    if "Overall" in results:
        r = results["Overall"]
        print(f"\n  {'Overall':<27s} {r['agreement']:>5.1f}%  ({r['correct']}/{r['total']})")

    if "_missing" in results:
        print(f"\n  ⚠  {results['_missing']} instances skipped (missing predictions)")

    print(f"{'=' * 62}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate metric agreement with human preference on VAQ-Bench."
    )
    parser.add_argument(
        "--prediction-file",
        required=True,
        help="Path to prediction JSON file (metric scores per candidate).",
    )
    parser.add_argument(
        "--benchmark-file",
        default="data/VAQ_Bench.json",
        help="Path to VAQ-Bench JSON file (default: data/VAQ_Bench.json).",
    )
    parser.add_argument(
        "--metric-name",
        default=None,
        help="Display name of the metric (for printing).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional: save results as JSON to this path.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_file)
    predictions = load_predictions(args.prediction_file)

    metric_name = args.metric_name or args.prediction_file
    results = compute_agreement(benchmark, predictions)
    print_results(results, metric_name)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

