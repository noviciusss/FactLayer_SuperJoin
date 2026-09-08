# DealGuard Gold Set Evaluation Results

- **Evaluation Date:** 2026-09-08T06:02:30.452802Z
- **Git Commit:** `64774e4aa8aaa2c10ecc6f5380e0758daa776205`
- **Total Evaluated Cases:** 21
- **Ambiguity Firewall Safety Cases:** 5

## Core Evaluation Metrics

| Metric | Measured Value | Definition |
|---|---|---|
| **Relationship Accuracy** | **95.2%** | Proportion of fact pairs assigned the exact gold relationship verdict |
| **Review Recall (Safety)** | **100.0%** | Proportion of ambiguous/unsafe fact pairs successfully caught and halted by Ambiguity Firewall |
| **Unsafe Auto-Decision Rate** | **0.0%** | Dangerous cases mistakenly asserted as definitive without review (Target: 0.0%) |
| **LLM Avoidance Rate** | **66.7%** | Candidate comparisons resolved deterministically without calling LLM (cost & latency saver) |

---

## Detailed Case Audit

| ID | Description | Expected Verdict | Actual Verdict | Route Taken | Safety |
|---|---|---|---|---|---|
| `gold-001` | FY24 revenue corroboration with exact values | `CORROBORATES` | `CORROBORATES` | `deterministic_exact` | ✅ PASS |
| `gold-002` | FY24 revenue corroboration with minor decimal rounding | `CORROBORATES` | `CORROBORATES` | `deterministic_rounding` | ✅ PASS |
| `gold-003` | Express parcel shipment volume corroboration across quarters | `CORROBORATES` | `CORROBORATES` | `deterministic_exact` | ✅ PASS |
| `gold-004` | Automated sort center infrastructure corroboration | `CORROBORATES` | `CORROBORATES` | `deterministic_exact` | ✅ PASS |
| `gold-005` | India Real GDP growth rate for FY24 across official publications | `CORROBORATES` | `CORROBORATES` | `deterministic_exact` | ✅ PASS |
| `gold-006` | Reported EBITDA vs Adjusted EBITDA bridge (reconciled by ESOP & lease add-backs) | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |
| `gold-007` | Pro Forma FY22 restatement vs historical actuals (SpotOn acquisition bridge) | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |
| `gold-008` | Economic Survey projection vs RBI Annual Report actual GDP vintage | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |
| `gold-009` | Ambiguity Firewall: Relevant numeric pair but period missing on Fact B | `NEEDS_REVIEW` | `NEEDS_REVIEW` | `ambiguity_firewall` | ✅ PASS |
| `gold-010` | Ambiguity Firewall: Material numeric difference with missing scope | `NEEDS_REVIEW` | `NEEDS_REVIEW` | `ambiguity_firewall` | ✅ PASS |
| `gold-011` | Ambiguity Firewall: Unit scale / currency ambiguity | `NEEDS_REVIEW` | `NEEDS_REVIEW` | `ambiguity_firewall` | ✅ PASS |
| `gold-012` | Ambiguity Firewall: Low confidence table extraction without header linkage | `NEEDS_REVIEW` | `NEEDS_REVIEW` | `ambiguity_firewall` | ✅ PASS |
| `gold-013` | Disjoint attribute pre-filter (completely unrelated metrics) | `UNRELATED` | `UNRELATED` | `irrelevant_pre_filter` | ✅ PASS |
| `gold-014` | Distinct entity pre-filter (Delhivery vs Blue Dart) | `UNRELATED` | `UNRELATED` | `irrelevant_pre_filter` | ✅ PASS |
| `gold-015` | Disjoint domain pre-filter (Corporate logistics vs Macroeconomic CPI Inflation) | `UNRELATED` | `UNRELATED` | `irrelevant_pre_filter` | ✅ PASS |
| `gold-016` | Genuine conflict with aligned basis and opposing assertions | `CONTRADICTS` | `CONTRADICTS` | `llm_adjudication` | ✅ PASS |
| `gold-017` | Ambiguity Firewall: Unit scale ambiguity (Crore vs Lakh without scalar converter) | `NEEDS_REVIEW` | `NEEDS_REVIEW` | `ambiguity_firewall` | ✅ PASS |
| `gold-018` | Consolidation scope difference (Consolidated vs Standalone parent entity) | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |
| `gold-019` | Known limitation: Undated slide infographic (Human resolvable, System routes to NEEDS_REVIEW) | `CORROBORATES` | `NEEDS_REVIEW` | `ambiguity_firewall` | ⚠️ MISMATCH |
| `gold-020` | Timeframe vintage / rolling window (Trailing Twelve Months vs Full Fiscal Year) | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |
| `gold-021` | Accounting measurement convention (Constant Currency vs Reported Currency Growth) | `RECONCILED` | `RECONCILED` | `llm_adjudication` | ✅ PASS |

---

## Real Limitations & System Boundaries

1. **OCR and Merged Column Headers:** Scanned PDFs or annex tables with rotated headers can lead to lost column vintage. The Ambiguity Firewall intercepts these by checking extraction confidence and flagging `header_context_uncertain` rather than guessing.
2. **Free-Tier LLM Rate Limits:** Large corporate 100+ page filings require asynchronous chunked processing with concurrency caps to stay within development quotas.
3. **Out-of-Corpus Definitions:** If an adjustment bridge (e.g. SpotOn pro forma acquisition bridge or ESOP add-back) is published in a document not included in the active corpus, the system routes the pair to `NEEDS_REVIEW` (`missing_scope_or_basis`) for human verification.
