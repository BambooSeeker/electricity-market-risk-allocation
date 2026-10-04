from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "c74_submission_readiness_gap_closure_package"
OUT.mkdir(exist_ok=True)


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def delta_row(df: pd.DataFrame, scope: str, challenger: str, reference: str) -> dict:
    a = df[(df["scope"] == scope) & (df["display_name"] == challenger)].iloc[0]
    b = df[(df["scope"] == scope) & (df["display_name"] == reference)].iloc[0]
    return {
        "scope": scope,
        "comparison": f"{challenger} vs {reference}",
        "delta_precision": a["precision"] - b["precision"],
        "delta_recall": a["recall"] - b["recall"],
        "delta_excess_share": a["excess_share"] - b["excess_share"],
        "delta_false_alert_burden": a["false_alert_burden"] - b["false_alert_burden"],
        "delta_gate_score": a["gate_score"] - b["gate_score"],
        "interpretation": "",
    }


def main() -> None:
    main_results = read_csv("c72_pre_manuscript_result_freeze_package/c72_frozen_main_result_table.csv")
    c73_scores = read_csv("c73_strict_reviewer_claim_defense_audit/c73_strict_scorecard.csv")
    c73_claims = read_csv("c73_strict_reviewer_claim_defense_audit/c73_claim_contribution_ledger.csv")

    external_validity = pd.DataFrame(
        [
            {
                "market_or_region": "China national spot-market reform / provincial pilots",
                "why_relevant": "China's spot-market design uses day-ahead and real-time clearing/settlement logic, so Zhejiang can be positioned as one provincial representative case rather than a universal proof.",
                "supports_claim": "Representative emerging regional market setting",
                "claim_boundary": "Do not claim all Chinese or global markets behave like Zhejiang.",
                "source_type": "official policy / regulator",
                "source_url": "https://www.ndrc.gov.cn/",
            },
            {
                "market_or_region": "Zhejiang provincial electricity spot market",
                "why_relevant": "The empirical case has half-hour market data and congestion-related price components, matching the paper's fixed-budget DA-RT spread-risk setting.",
                "supports_claim": "Primary case study",
                "claim_boundary": "Single-market evidence; external transfer requires similar DA/RT and congestion/risk information structures.",
                "source_type": "project data + provincial mechanism documents",
                "source_url": "local project data",
            },
            {
                "market_or_region": "CAISO",
                "why_relevant": "Has day-ahead and real-time market operations and congestion/location-sensitive price formation, making the DA/RT spread-risk problem conceptually transferable.",
                "supports_claim": "Comparable mature-market mechanism class",
                "claim_boundary": "Mature nodal market; not an emerging regional-market proof.",
                "source_type": "market operator",
                "source_url": "https://www.caiso.com/",
            },
            {
                "market_or_region": "PJM",
                "why_relevant": "Operates day-ahead and real-time energy markets with LMP-based settlement, showing the DA/RT distinction is not Zhejiang-specific.",
                "supports_claim": "Comparable DA/RT market architecture",
                "claim_boundary": "Different market maturity, geography, and participants.",
                "source_type": "market operator",
                "source_url": "https://www.pjm.com/",
            },
            {
                "market_or_region": "ERCOT",
                "why_relevant": "Operates day-ahead and real-time markets; fixed-budget risk screening can be motivated in markets with volatile DA/RT deviations.",
                "supports_claim": "Comparable DA/RT risk-screening motivation",
                "claim_boundary": "ERCOT's scarcity pricing and market design differ; use only as mechanism analogy.",
                "source_type": "market operator",
                "source_url": "https://www.ercot.com/",
            },
        ]
    )

    method_family = pd.DataFrame(
        [
            {
                "family": "Base distributional tail-risk ranking",
                "representative_methods": "C20/C20b",
                "paper_role": "Base operational ranking baseline",
                "main_or_appendix": "Main text",
                "scientific_role": "Establishes a nontrivial probabilistic/risk-score baseline.",
                "do_not_claim": "Do not call it the final contribution.",
            },
            {
                "family": "Boundary-aware fixed-budget ranking",
                "representative_methods": "C44",
                "paper_role": "Robust conservative anchor",
                "main_or_appendix": "Main text",
                "scientific_role": "Validation-frozen improvement under the target fixed-budget task.",
                "do_not_claim": "Do not claim large universal improvement.",
            },
            {
                "family": "Deep-sequence enhanced fixed-budget candidate",
                "representative_methods": "C50",
                "paper_role": "Top30 enhanced candidate",
                "main_or_appendix": "Main text with strict boundary",
                "scientific_role": "Best current top30 fixed-budget enhancement.",
                "do_not_claim": "Do not claim all-budget dominance or frontier model invention.",
            },
            {
                "family": "Post-processing / conformal / isotonic-style probabilistic baselines",
                "representative_methods": "C31/C62",
                "paper_role": "Modern probabilistic credibility baseline",
                "main_or_appendix": "Compact main reference + appendix",
                "scientific_role": "Shows probability/distributional evaluation is considered.",
                "do_not_claim": "Do not replace fixed-budget metrics with calibration metrics.",
            },
            {
                "family": "Generative / diffusion-style risk baseline",
                "representative_methods": "C33",
                "paper_role": "Representative generative baseline",
                "main_or_appendix": "Appendix or compact baseline table",
                "scientific_role": "Defends against omission of generative probabilistic methods.",
                "do_not_claim": "Do not call it a frontier diffusion model.",
            },
            {
                "family": "State-space / selective-sequence baseline",
                "representative_methods": "C32/C59",
                "paper_role": "Modern sequence audit",
                "main_or_appendix": "Appendix",
                "scientific_role": "Checks whether modern sequence structure directly replaces the fixed-budget rule.",
                "do_not_claim": "Do not call it official Mamba/S4 unless implemented as such.",
            },
            {
                "family": "Multi-scale deep temporal baseline",
                "representative_methods": "C60/C61",
                "paper_role": "Deep sequence robustness audit",
                "main_or_appendix": "Appendix",
                "scientific_role": "Tests TCN/attention-style sequence learning under the risk task.",
                "do_not_claim": "Do not keep adding variants after no robust main gain.",
            },
            {
                "family": "Mechanism heterogeneity diagnostics",
                "representative_methods": "C65/C68/C69/C70",
                "paper_role": "Stress-test and mechanism appendix",
                "main_or_appendix": "Appendix + short discussion",
                "scientific_role": "Explains pressure-month and renewable-state heterogeneity.",
                "do_not_claim": "Do not promote failed C70 allocation rule.",
            },
        ]
    )

    scopes = ["all_available_months", "holdout_pressure", "non_pressure_available_months"]
    rows = []
    base = "Base distributional tail-risk ranking"
    c44 = "C44 robust conservative anchor"
    c50 = "C50 deep-sequence top30 enhanced candidate"
    for scope in scopes:
        rows.append(delta_row(main_results, scope, c44, base))
        rows.append(delta_row(main_results, scope, c50, base))
        rows.append(delta_row(main_results, scope, c50, c44))
    effect = pd.DataFrame(rows)
    for i, row in effect.iterrows():
        if row["delta_excess_share"] > row["delta_recall"]:
            text = "Improvement is driven more by covered loss magnitude than by event-count recall."
        elif row["delta_recall"] > 0:
            text = "Improvement includes event-count recall gain."
        else:
            text = "No event-count recall gain; interpret only through ExcessShare/false-alert changes."
        if row["scope"] == "holdout_pressure":
            text += " Pressure-scope claim must remain stress-test bounded."
        effect.loc[i, "interpretation"] = text

    journal_safety = pd.DataFrame(
        [
            {
                "target_level": "Applied Energy / similarly selective top energy journal",
                "current_safety": "not_safe_yet",
                "estimated_score_after_c74": 8.2,
                "main_blocker": "Single-market external validity and modest main effect size.",
                "what_would_raise_confidence": "Additional market/data evidence or clearly stronger robust effect without post-hoc tuning.",
            },
            {
                "target_level": "Strong Q1 power/energy systems journal",
                "current_safety": "possible_with_careful_positioning",
                "estimated_score_after_c74": 8.35,
                "main_blocker": "Need clean story and compact evidence, not overclaiming.",
                "what_would_raise_confidence": "Finalize claim ledger and write bounded mechanism-case framing.",
            },
            {
                "target_level": "Broader engineering/application journal",
                "current_safety": "more_comfortable",
                "estimated_score_after_c74": 8.5,
                "main_blocker": "Less novelty pressure, but still needs clear practical value.",
                "what_would_raise_confidence": "Strong application framing and transparent appendix.",
            },
        ]
    )

    c74_decision = pd.DataFrame(
        [
            {
                "question": "Did C74 solve external validity?",
                "answer": "partially",
                "reason": "It creates a market-scope boundary and comparison table, but does not add another empirical market.",
                "score_effect": "Raises writing defensibility, not empirical generality.",
            },
            {
                "question": "Did C74 solve method-modernity concerns?",
                "answer": "partially",
                "reason": "It organizes existing modern baselines into a coherent family map; it does not create a new winning frontier model.",
                "score_effect": "Improves reviewer clarity and reduces clutter risk.",
            },
            {
                "question": "Did C74 solve modest effect-size concerns?",
                "answer": "partially",
                "reason": "It decomposes effects into Recall/ExcessShare/FalseAlertBurden, showing the main gain is often magnitude coverage.",
                "score_effect": "Improves interpretability, not raw performance.",
            },
            {
                "question": "Can full manuscript writing start now?",
                "answer": "not_final_writing",
                "reason": "A controlled outline can start, but final writing still needs either acceptance of the target-level risk or new evidence.",
                "score_effect": "Keeps the project honest.",
            },
        ]
    )

    score_update = pd.DataFrame(
        [
            {"dimension": "Problem novelty and framing", "before_c74": 8.4, "after_c74": 8.45, "basis": "External boundary and effect framing clarified."},
            {"dimension": "Main result strength", "before_c74": 7.9, "after_c74": 8.0, "basis": "Effect decomposition improves interpretation but not raw gain."},
            {"dimension": "Metric validity", "before_c74": 8.5, "after_c74": 8.5, "basis": "Already strong; unchanged."},
            {"dimension": "Method modernity defense", "before_c74": 8.0, "after_c74": 8.25, "basis": "Method-family map reduces scattered-baseline risk."},
            {"dimension": "External validity", "before_c74": 7.6, "after_c74": 7.85, "basis": "Market-scope table helps, but no second empirical market."},
            {"dimension": "Practical value", "before_c74": 8.3, "after_c74": 8.35, "basis": "Fixed-budget operational interpretation clarified."},
            {"dimension": "Robustness and anti-overfitting discipline", "before_c74": 8.1, "after_c74": 8.15, "basis": "No-go rules retained; no new tuning."},
        ]
    )
    overall_before = score_update["before_c74"].mean()
    overall_after = score_update["after_c74"].mean()

    report = f"""# C74 Submission-Readiness Gap Closure Package

## Purpose

C74 consolidates four scattered but essential reviewer-defense tasks:

1. external validity and market-scope boundary;
2. method-family clarity;
3. component-level effect decomposition;
4. journal-target safety.

It does **not** add a new model and does **not** raise scores by wording alone.

## Result

Strict average before C74: {overall_before:.2f}

Strict average after C74: {overall_after:.2f}

The project is more coherent after C74, but it is still not safely above 8.5 for an Applied Energy-level target.

## What C74 Actually Improves

- External validity is reframed from "Zhejiang proves everything" to "Zhejiang is a representative regional-market case under specified transfer conditions".
- Method families are organized so the paper does not look like scattered model stacking.
- Effects are decomposed into Recall, ExcessShare, FalseAlertBurden, and GateScore.
- Journal safety is separated from wishful thinking.

## Remaining Hard Truth

C74 improves defensibility, not raw empirical strength. Without either a second empirical market, stronger robust effect size, or a lower target-journal risk appetite, the project remains promising but not 8.5+ safe.

## Recommended Next Step

C75 should be a binary decision:

- either accept the current evidence package and start a controlled manuscript outline for a strong-but-bounded target;
- or seek one concrete evidence upgrade, most likely external data/literature-supported market-scope evidence, before final writing.
"""

    external_validity.to_csv(OUT / "c74_external_validity_market_scope_table.csv", index=False, encoding="utf-8-sig")
    method_family.to_csv(OUT / "c74_method_family_clarity_table.csv", index=False, encoding="utf-8-sig")
    effect.to_csv(OUT / "c74_component_effect_decomposition.csv", index=False, encoding="utf-8-sig")
    journal_safety.to_csv(OUT / "c74_journal_target_safety_matrix.csv", index=False, encoding="utf-8-sig")
    c74_decision.to_csv(OUT / "c74_gap_closure_decision_table.csv", index=False, encoding="utf-8-sig")
    score_update.to_csv(OUT / "c74_score_update_without_new_experiment.csv", index=False, encoding="utf-8-sig")
    c73_scores.to_csv(OUT / "c74_imported_c73_strict_scorecard.csv", index=False, encoding="utf-8-sig")
    c73_claims.to_csv(OUT / "c74_imported_c73_claim_ledger.csv", index=False, encoding="utf-8-sig")
    (OUT / "c74_submission_readiness_gap_closure_report.md").write_text(report, encoding="utf-8")

    print(f"C74 strict average before: {overall_before:.2f}")
    print(f"C74 strict average after: {overall_after:.2f}")
    print("Wrote C74 outputs:")
    for p in sorted(OUT.iterdir()):
        print(f"- {p}")


if __name__ == "__main__":
    main()
