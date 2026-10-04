from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "c72_pre_manuscript_result_freeze_package"
OUT.mkdir(exist_ok=True)


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    c54_metrics = read_csv("c54_top30_final_robustness/c54_top30_main_metrics.csv")
    c54_stability = read_csv("c54_top30_final_robustness/c54_top30_stability_scorecard.csv")
    c55_baselines = read_csv("c55_baseline_compression_table/c55_main_text_compressed_baselines.csv")
    c62_quantile = read_csv("c62_calibration_distributional_audit/c62_quantile_distributional_rank.csv")
    c66_claims = read_csv("c66_appendix_budget_traceability_audit/c66_claim_audit_scorecard.csv")
    c70_decision = read_csv("c70_renewable_stratified_allocation_rule/c70_go_no_go_decision.csv")
    c71_stop_go = read_csv("c71_final_experiment_stop_go_audit/c71_experiment_stop_go_decision.csv")
    c71_readiness = read_csv("c71_final_experiment_stop_go_audit/c71_manuscript_readiness_scorecard.csv")
    c71_placement = read_csv("c71_final_experiment_stop_go_audit/c71_evidence_placement_map.csv")

    main_methods = [
        "C20b_base",
        "C44_limited_budget_band_until30",
        "c22_deep_w20_until30_else_c44",
    ]
    scopes = ["all_available_months", "holdout_pressure", "non_pressure_available_months"]
    main = c54_metrics[
        c54_metrics["method"].isin(main_methods)
        & c54_metrics["scope"].isin(scopes)
        & c54_metrics["budget"].eq(0.3)
    ].copy()
    labels = {
        "C20b_base": "Base distributional tail-risk ranking",
        "C44_limited_budget_band_until30": "C44 robust conservative anchor",
        "c22_deep_w20_until30_else_c44": "C50 deep-sequence top30 enhanced candidate",
    }
    main["display_name"] = main["method"].map(labels)
    cols = [
        "scope",
        "display_name",
        "events",
        "alerts",
        "precision",
        "recall",
        "excess_share",
        "false_alert_burden",
        "gate_score",
        "delta_gate_vs_base",
        "delta_gate_vs_c44",
    ]
    main_table = main[cols].sort_values(["scope", "display_name"])

    metric_ledger = pd.DataFrame(
        [
            {
                "metric": "Recall",
                "main_use": "How many true lower-tail risk events are captured under a fixed alert budget.",
                "must_report": "Yes",
                "claim_boundary": "Event-count coverage only; it does not measure loss magnitude.",
            },
            {
                "metric": "ExcessShare",
                "main_use": "How much negative excess spread magnitude is covered under the same budget.",
                "must_report": "Yes",
                "claim_boundary": "Magnitude coverage; can improve even when event recall is unchanged.",
            },
            {
                "metric": "Precision",
                "main_use": "How many selected alerts are true lower-tail events.",
                "must_report": "Yes",
                "claim_boundary": "Budget-dependent; should not be compared without the same budget.",
            },
            {
                "metric": "FalseAlertBurden",
                "main_use": "Fraction of selected alerts that are not true lower-tail events.",
                "must_report": "Yes",
                "claim_boundary": "A burden indicator, not a standalone utility or trading loss.",
            },
            {
                "metric": "GateScore",
                "main_use": "Auxiliary compact screening score: Recall + ExcessShare - 0.2*FalseAlertBurden.",
                "must_report": "Auxiliary only",
                "claim_boundary": "Never use alone as the final performance proof.",
            },
            {
                "metric": "AUC / PR-AUC",
                "main_use": "Global ranking diagnostic.",
                "must_report": "Appendix or secondary",
                "claim_boundary": "Not the primary metric because operation uses fixed budgets.",
            },
            {
                "metric": "Pinball / calibration error",
                "main_use": "Distributional/probabilistic quality audit.",
                "must_report": "Appendix or distributional subsection",
                "claim_boundary": "Useful for modern-model defense, not a substitute for fixed-budget coverage.",
            },
        ]
    )

    no_go = c71_stop_go[c71_stop_go["decision"].isin(["STOP", "STOP_OR_PAUSE"])].copy()
    no_go["freeze_status"] = "Do not resurrect without new data or a pre-registered reason."

    appendix_map = pd.concat(
        [
            c71_placement.assign(source="C71 placement map"),
            pd.DataFrame(
                [
                    {
                        "evidence_block": "C62 calibration/distributional audit",
                        "placement": "Appendix with compact main-text reference",
                        "claim_strength": "Modern probabilistic defense",
                        "claim_boundary": "Does not replace fixed-budget lower-tail metrics.",
                        "source": "C62",
                    },
                    {
                        "evidence_block": "C55 compressed representative baselines",
                        "placement": "Main baseline table plus appendix variants",
                        "claim_strength": "Baseline completeness",
                        "claim_boundary": "Avoid presenting every one-off variant in main text.",
                        "source": "C55",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    gap_list = pd.DataFrame(
        [
            {
                "gap": "Final claim ledger",
                "status": "Needed before writing",
                "why_it_matters": "Prevents overclaiming C50, pressure months, and renewable strata.",
                "next_action": "C73 should convert C72 tables into claim-by-claim referee defense.",
            },
            {
                "gap": "Figure/table shortlist",
                "status": "Needed before writing",
                "why_it_matters": "The paper cannot include every experiment; excess evidence will look unfocused.",
                "next_action": "Freeze 3-4 main tables/figures and move the rest to appendix.",
            },
            {
                "gap": "Journal-target readiness",
                "status": "Still not 8.5+ safe",
                "why_it_matters": "Applied Energy-level confidence needs stronger external validity or more decisive robustness.",
                "next_action": "Run one final strict reviewer gate after result package freeze.",
            },
            {
                "gap": "Data-scope boundary",
                "status": "Needs explicit phrasing",
                "why_it_matters": "Zhejiang-only data supports a representative case, not a universal market theorem.",
                "next_action": "Write a bounded mechanism statement, not a broad global claim.",
            },
        ]
    )

    selected_summary = {
        "c44_vs_base_all_top30_delta_gate": c54_stability.loc[
            c54_stability["comparison"].eq("C44_vs_base"), "all_top30_delta_gate"
        ].iloc[0],
        "c50_vs_c44_all_top30_delta_gate": c54_stability.loc[
            c54_stability["comparison"].eq("C50_vs_C44"), "all_top30_delta_gate"
        ].iloc[0],
        "c44_vs_base_pressure_delta_gate": c54_stability.loc[
            c54_stability["comparison"].eq("C44_vs_base"), "pressure_top30_delta_gate"
        ].iloc[0],
        "c50_vs_c44_pressure_delta_gate": c54_stability.loc[
            c54_stability["comparison"].eq("C50_vs_C44"), "pressure_top30_delta_gate"
        ].iloc[0],
        "c70_holdout_delta_gate_vs_c50": c70_decision["holdout_delta_gate_vs_c50"].iloc[0],
        "conservative_overall_score": c71_readiness["score"].mean(),
    }

    report = f"""# C72 Pre-Manuscript Result Freeze Package

## Decision

The project is **not yet entering full manuscript writing**. It is entering the final pre-manuscript freeze stage.

The current result package is usable, but the safest next step is one final strict claim-defense audit before drafting the paper.

## Frozen Main Result Candidates

The main empirical table should center on:

1. Base distributional tail-risk ranking;
2. C44 robust conservative anchor;
3. C50 deep-sequence top30 enhanced candidate.

Key deltas:

- C44 vs base, all-month top30 delta GateScore: {fmt(selected_summary['c44_vs_base_all_top30_delta_gate'])}
- C50 vs C44, all-month top30 delta GateScore: {fmt(selected_summary['c50_vs_c44_all_top30_delta_gate'])}
- C44 vs base, pressure top30 delta GateScore: {fmt(selected_summary['c44_vs_base_pressure_delta_gate'])}
- C50 vs C44, pressure top30 delta GateScore: {fmt(selected_summary['c50_vs_c44_pressure_delta_gate'])}

These are meaningful but not large enough to support exaggerated claims.

## Frozen Boundary

- Primary application: fixed-budget lower-tail DA-RT spread risk coverage.
- Primary budget: top30.
- Primary evidence: Recall, ExcessShare, Precision/FalseAlertBurden.
- GateScore: auxiliary screening score only.
- Pressure months: stress test, not solved problem.
- Renewable strata: mechanism diagnostic, not allocation rule.
- C70-like boost rules: stopped because holdout failed.

## Main Text vs Appendix

Main text should be lean:

- problem definition and metric rationale;
- compact baseline table;
- C44/C50 top30 result table;
- short robustness summary;
- mechanism/stress-test interpretation.

Appendix should absorb:

- multi-budget sensitivity;
- bootstrap/month stress audits;
- modern baseline and calibration audits;
- renewable strata diagnostics;
- C70 negative allocation result;
- no-go experiment ledger.

## Remaining Before Writing

C73 should be a final strict reviewer gate:

- check whether every claimed contribution has a corresponding table;
- check whether every weak result is bounded honestly;
- check whether the manuscript can survive the questions: novelty, modernity, external validity, and practical value.
"""

    main_table.to_csv(OUT / "c72_frozen_main_result_table.csv", index=False, encoding="utf-8-sig")
    metric_ledger.to_csv(OUT / "c72_metric_claim_ledger.csv", index=False, encoding="utf-8-sig")
    no_go.to_csv(OUT / "c72_no_go_experiment_list.csv", index=False, encoding="utf-8-sig")
    appendix_map.to_csv(OUT / "c72_appendix_evidence_map.csv", index=False, encoding="utf-8-sig")
    gap_list.to_csv(OUT / "c72_remaining_gap_list.csv", index=False, encoding="utf-8-sig")
    c66_claims.to_csv(OUT / "c72_imported_c66_claim_audit.csv", index=False, encoding="utf-8-sig")
    c55_baselines.to_csv(OUT / "c72_imported_c55_main_baselines.csv", index=False, encoding="utf-8-sig")
    c62_quantile.to_csv(OUT / "c72_imported_c62_distributional_rank.csv", index=False, encoding="utf-8-sig")
    (OUT / "c72_pre_manuscript_result_freeze_report.md").write_text(report, encoding="utf-8")

    print("Wrote C72 outputs:")
    for p in sorted(OUT.iterdir()):
        print(f"- {p}")


if __name__ == "__main__":
    main()
