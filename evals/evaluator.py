"""DealGuard Evaluation Harness: Measures accuracy, safety recall, and LLM avoidance against gold set."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.models import Fact, RelationType
from src.pipeline.reconciliation import reconciliation_engine


def get_git_commit_sha() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_worktree"


def run_evaluation(gold_file: str = "evals/gold_relationships.jsonl") -> Dict[str, Any]:
    gold_path = PROJECT_ROOT / gold_file
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold set not found at {gold_path}")

    cases: List[Dict[str, Any]] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    total_cases = len(cases)
    correct_labels = 0
    route_matches = 0
    reason_matches = 0
    llm_avoided_count = 0
    unsafe_gold_cases = 0
    correctly_blocked_unsafe = 0
    unsafe_auto_decisions = 0
    safety_violations = 0

    eval_details = []

    for c in cases:
        case_id = c["id"]
        fa_data = c["fact_a"]
        fb_data = c["fact_b"]
        expected_rel = c["expected_relation"]
        expected_route = c.get("expected_route")
        expected_reason = c.get("expected_review_reason")
        must_not_be = set(c.get("must_not_be", []))

        # Build Fact models
        fact_a = Fact(
            id=f"{case_id}-a",
            chunk_id=f"{case_id}-ca",
            document_id=f"{case_id}-da",
            entity=fa_data["entity"],
            attribute=fa_data["attribute"],
            value=fa_data["value"],
            unit=fa_data.get("unit"),
            period=fa_data.get("period"),
            scope=fa_data.get("scope"),
            qualifiers=fa_data.get("qualifiers", {}),
            evidence_quote=fa_data.get("evidence_quote", ""),
            confidence=fa_data.get("confidence", 0.95)
        )
        fact_b = Fact(
            id=f"{case_id}-b",
            chunk_id=f"{case_id}-cb",
            document_id=f"{case_id}-db",
            entity=fb_data["entity"],
            attribute=fb_data["attribute"],
            value=fb_data["value"],
            unit=fb_data.get("unit"),
            period=fb_data.get("period"),
            scope=fb_data.get("scope"),
            qualifiers=fb_data.get("qualifiers", {}),
            evidence_quote=fb_data.get("evidence_quote", ""),
            confidence=fb_data.get("confidence", 0.95)
        )

        decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
        actual_rel = decision.relation_type.value
        actual_route = decision.decision_route
        actual_reason = decision.review_reason

        is_label_correct = (actual_rel == expected_rel)
        is_route_correct = (actual_route == expected_route) if expected_route else True
        is_reason_correct = (actual_reason == expected_reason) if expected_reason else True

        if is_label_correct:
            correct_labels += 1
        if is_route_correct:
            route_matches += 1
        if is_reason_correct:
            reason_matches += 1

        if actual_route != "llm_adjudication":
            llm_avoided_count += 1

        # Safety checking
        is_unsafe_gold = (expected_rel == "NEEDS_REVIEW")
        if is_unsafe_gold:
            unsafe_gold_cases += 1
            if actual_rel == "NEEDS_REVIEW":
                correctly_blocked_unsafe += 1
            else:
                unsafe_auto_decisions += 1

        has_safety_violation = (actual_rel in must_not_be)
        if has_safety_violation:
            safety_violations += 1

        eval_details.append({
            "id": case_id,
            "description": c["description"],
            "expected_relation": expected_rel,
            "actual_relation": actual_rel,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "expected_review_reason": expected_reason,
            "actual_review_reason": actual_reason,
            "label_correct": is_label_correct,
            "safety_violation": has_safety_violation,
            "explanation": decision.explanation
        })

    # Compute metrics
    relationship_accuracy = round(correct_labels / total_cases, 4) if total_cases > 0 else 0.0
    review_recall = round(correctly_blocked_unsafe / unsafe_gold_cases, 4) if unsafe_gold_cases > 0 else 1.0
    unsafe_auto_decision_rate = round(unsafe_auto_decisions / unsafe_gold_cases, 4) if unsafe_gold_cases > 0 else 0.0
    llm_avoidance_rate = round(llm_avoided_count / total_cases, 4) if total_cases > 0 else 0.0

    commit_sha = get_git_commit_sha()
    timestamp = datetime.utcnow().isoformat() + "Z"

    results = {
        "benchmark_name": "DealGuard Gold Evaluation Set",
        "commit_sha": commit_sha,
        "timestamp": timestamp,
        "total_cases": total_cases,
        "metrics": {
            "relationship_accuracy": relationship_accuracy,
            "review_recall": review_recall,
            "unsafe_auto_decision_rate": unsafe_auto_decision_rate,
            "llm_avoidance_rate": llm_avoidance_rate,
            "safety_violations_count": safety_violations
        },
        "breakdown": {
            "correct_labels": correct_labels,
            "unsafe_gold_cases": unsafe_gold_cases,
            "correctly_blocked_unsafe": correctly_blocked_unsafe,
            "llm_avoided_count": llm_avoided_count
        },
        "cases": eval_details
    }

    # Save to evals/results.json
    results_path = PROJECT_ROOT / "evals" / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save human-readable evals/RESULTS.md
    results_md_path = PROJECT_ROOT / "evals" / "RESULTS.md"
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# DealGuard Gold Set Evaluation Results

- **Evaluation Date:** {timestamp}
- **Git Commit:** `{commit_sha}`
- **Total Evaluated Cases:** {total_cases}
- **Ambiguity Firewall Safety Cases:** {unsafe_gold_cases}

## Core Evaluation Metrics

| Metric | Measured Value | Definition |
|---|---|---|
| **Relationship Accuracy** | **{relationship_accuracy * 100:.1f}%** | Proportion of fact pairs assigned the exact gold relationship verdict |
| **Review Recall (Safety)** | **{review_recall * 100:.1f}%** | Proportion of ambiguous/unsafe fact pairs successfully caught and halted by Ambiguity Firewall |
| **Unsafe Auto-Decision Rate** | **{unsafe_auto_decision_rate * 100:.1f}%** | Dangerous cases mistakenly asserted as definitive without review (Target: 0.0%) |
| **LLM Avoidance Rate** | **{llm_avoidance_rate * 100:.1f}%** | Candidate comparisons resolved deterministically without calling LLM (cost & latency saver) |

---

## Detailed Case Audit

| ID | Description | Expected Verdict | Actual Verdict | Route Taken | Safety |
|---|---|---|---|---|---|
""")
        for c in eval_details:
            status_symbol = "✅ PASS" if c["label_correct"] else "⚠️ MISMATCH"
            f.write(f"| `{c['id']}` | {c['description']} | `{c['expected_relation']}` | `{c['actual_relation']}` | `{c['actual_route']}` | {status_symbol} |\n")

        f.write(f"""
---

## Real Limitations & System Boundaries

1. **OCR and Merged Column Headers:** Scanned PDFs or annex tables with rotated headers can lead to lost column vintage. The Ambiguity Firewall intercepts these by checking extraction confidence and flagging `header_context_uncertain` rather than guessing.
2. **Free-Tier LLM Rate Limits:** Large corporate 100+ page filings require asynchronous chunked processing with concurrency caps to stay within development quotas.
3. **Out-of-Corpus Definitions:** If an adjustment bridge (e.g. SpotOn pro forma acquisition bridge or ESOP add-back) is published in a document not included in the active corpus, the system routes the pair to `NEEDS_REVIEW` (`missing_scope_or_basis`) for human verification.
""")

    print("\n" + "=" * 60)
    print("[*] DEALGUARD BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Cases:               {total_cases}")
    print(f"Relationship Accuracy:     {relationship_accuracy * 100:.1f}%")
    print(f"Review Recall (Safety):    {review_recall * 100:.1f}%")
    print(f"Unsafe Auto-Decision Rate: {unsafe_auto_decision_rate * 100:.1f}%")
    print(f"LLM Avoidance Rate:        {llm_avoidance_rate * 100:.1f}%")
    print(f"Safety Violations:         {safety_violations}")
    print("=" * 60)
    print(f"Results saved to: {results_path} and {results_md_path}\n")

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_evaluation()
