from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "taskB_main_result_strengthening" / "c77_stress_gated_deep_boundary_enhancement" / "c77_candidate_screen.csv"
OUT = ROOT / "taskB_main_result_strengthening" / "c80_c77_selection_leakage_and_ablation_audit"
OUT.mkdir(parents=True, exist_ok=True)


def feature(method: str) -> str:
    body = method.removeprefix("C77_")
    for marker in ["_q60_", "_q70_", "_q80_"]:
        if marker in body:
            return body.split(marker)[0]
    return body


def parse_param(method: str, prefix: str) -> int:
    match = re.search(rf"_{prefix}(\d+)(?:_|$)", method)
    if not match:
        return -1
    return int(match.group(1))


def main() -> None:
    df = pd.read_csv(SRC)
    df["feature"] = df["method"].map(feature)
    df["q"] = df["method"].map(lambda x: parse_param(x, "q"))
    df["replace_pct"] = df["method"].map(lambda x: parse_param(x, "r"))
    df["pool_pct"] = df["method"].map(lambda x: parse_param(x, "pool"))

    val_pass = df[df["passes_validation_gate"].eq(True)].copy()
    val_only_best = val_pass.sort_values(
        ["val_delta_gate_vs_c44", "val_delta_recall_vs_c44", "val_delta_excess_vs_c44"],
        ascending=[False, False, False],
    ).head(10)

    conservative_val = val_pass[
        (val_pass["replace_pct"].eq(5))
        & (val_pass["pool_pct"].eq(50))
        & (val_pass["val_delta_recall_vs_c44"] >= 0)
        & (val_pass["val_delta_excess_vs_c44"] >= 0)
    ].copy()
    conservative_val = conservative_val.sort_values(
        ["val_delta_gate_vs_c44", "q", "feature"], ascending=[False, True, True]
    ).head(20)

    feature_summary = (
        df.groupby("feature", as_index=False)
        .agg(
            candidates=("method", "count"),
            val_pass_count=("passes_validation_gate", "sum"),
            holdout_safe_count=("passes_holdout_safety", "sum"),
            promotable_count=("promotable", "sum"),
            best_val_delta=("val_delta_gate_vs_c44", "max"),
            best_holdout_delta=("holdout_delta_gate_vs_c44", "max"),
            mean_holdout_delta=("holdout_delta_gate_vs_c44", "mean"),
        )
        .sort_values(["promotable_count", "best_holdout_delta", "best_val_delta"], ascending=[False, False, False])
    )

    # Current selected candidate from C77 is not validation-only best. Audit whether
    # the selection can be defended without holdout sorting.
    current = df[df["method"].eq("C77_net_load_pred_q60_r5_pool50")].copy()
    current_rank_by_val = int((df["val_delta_gate_vs_c44"] > current.iloc[0]["val_delta_gate_vs_c44"]).sum() + 1)
    current_rank_by_holdout = int((df["holdout_delta_gate_vs_c44"] > current.iloc[0]["holdout_delta_gate_vs_c44"]).sum() + 1)

    decision_rows = [
        {
            "issue": "holdout_selection_leakage",
            "status": "RISK_CONFIRMED",
            "evidence": "C77 selected candidate is rank "
            + str(current_rank_by_val)
            + " by validation delta but rank "
            + str(current_rank_by_holdout)
            + " by holdout delta.",
            "action": "Do not claim C77 was selected by a clean validation-only rule unless a validation-only rule is rebuilt.",
        },
        {
            "issue": "net_load_single_feature_dependence",
            "status": "PARTLY_DEFENSIBLE",
            "evidence": "Multiple feature families pass validation and promotion gates, so the idea is not unique to net_load_pred; however current selected candidate uses holdout-favorable net_load.",
            "action": "Rebuild C77 selection with validation-only or pre-specified parsimony before final manuscript use.",
        },
        {
            "issue": "ablation_structure",
            "status": "CLEAR",
            "evidence": "C44 is anchor, C50 is ungated deep complement, C77 is stress-gated boundary complement.",
            "action": "Use C44/C50/C77 as ordered ablation, but only after C77 selection rule is clean.",
        },
    ]
    decision = pd.DataFrame(decision_rows)

    val_only_best.to_csv(OUT / "c80_validation_only_best_candidates.csv", index=False, encoding="utf-8-sig")
    conservative_val.to_csv(OUT / "c80_conservative_validation_candidates.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(OUT / "c80_feature_family_summary.csv", index=False, encoding="utf-8-sig")
    current.to_csv(OUT / "c80_current_c77_candidate_audit.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "c80_selection_leakage_decision.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C80 C77 Selection Leakage and Ablation Audit",
        "",
        "Purpose: test whether C77 can be defended as a clean selected method rather than a holdout-favorable candidate.",
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False),
        "",
        "## Current Candidate",
        "",
        current.to_markdown(index=False),
        "",
        "## Validation-Only Best Candidates",
        "",
        val_only_best.to_markdown(index=False),
        "",
        "## Conservative Validation Candidates",
        "",
        conservative_val.to_markdown(index=False),
        "",
        "## Feature Family Summary",
        "",
        feature_summary.to_markdown(index=False),
        "",
    ]
    (OUT / "c80_c77_selection_leakage_and_ablation_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
