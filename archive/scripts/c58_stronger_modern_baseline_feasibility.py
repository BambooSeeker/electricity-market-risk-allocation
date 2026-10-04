from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c58_stronger_modern_baseline_feasibility"
OUT.mkdir(exist_ok=True)


def feasibility_table() -> pd.DataFrame:
    rows = [
        {
            "candidate": "Supervised Mamba/S4-style sequence baseline",
            "modern_family": "selective/state-space sequence model",
            "fit_to_our_task": "high",
            "why_it_fits": "It can consume multivariate exogenous market/mechanism features and produce tail-risk scores under the same rolling-month fixed-budget protocol.",
            "data_support": "medium",
            "implementation_effort": "medium_high",
            "a800_usefulness": "high",
            "expected_value": "high_if_dependency_available",
            "main_risk": "mamba_ssm/S4 dependencies may be hard; small evaluation window can still limit gains.",
            "paper_role_if_successful": "strong modern baseline or upgraded deep top30 candidate",
            "decision": "first_priority_feasibility_check",
        },
        {
            "candidate": "Chronos/TimesFM/Moirai-style foundation forecast baseline",
            "modern_family": "time-series foundation model",
            "fit_to_our_task": "medium",
            "why_it_fits": "It offers a 2024+ zero-shot/foundation-model comparison and can generate predictive distributions or quantiles from historical spreads.",
            "data_support": "medium",
            "implementation_effort": "medium",
            "a800_usefulness": "medium",
            "expected_value": "medium_high_as_baseline",
            "main_risk": "Most foundation models are weak at using our rich exogenous market/mechanism features; they may be strong modern baselines but not the best main method.",
            "paper_role_if_successful": "frontier-style external baseline, likely appendix or main baseline table",
            "decision": "second_priority_if_installable",
        },
        {
            "candidate": "Diffusion Transformer / TimeDART-style probabilistic baseline",
            "modern_family": "diffusion-transformer time-series generation",
            "fit_to_our_task": "medium_high",
            "why_it_fits": "It naturally generates scenario distributions for tail-risk coverage and aligns with probabilistic forecasting.",
            "data_support": "medium_low",
            "implementation_effort": "high",
            "a800_usefulness": "high",
            "expected_value": "medium",
            "main_risk": "Full transformer diffusion is complex; our current sample size may not justify a large generative model, and a simplified version may again look toy.",
            "paper_role_if_successful": "strong generative baseline only if implemented cleanly",
            "decision": "defer_unless_mamba_or_foundation_fails",
        },
        {
            "candidate": "KAN / KAN-Transformer baseline",
            "modern_family": "Kolmogorov-Arnold network",
            "fit_to_our_task": "medium",
            "why_it_fits": "KAN can model nonlinear feature interactions and may provide interpretable nonlinear risk scores.",
            "data_support": "medium",
            "implementation_effort": "medium",
            "a800_usefulness": "low_medium",
            "expected_value": "medium_low",
            "main_risk": "KAN is not as clearly tied to probabilistic sequence forecasting or tail-risk coverage; may look like architecture chasing.",
            "paper_role_if_successful": "auxiliary ablation, not main modern baseline",
            "decision": "not_priority",
        },
        {
            "candidate": "LLM-style prompt/time-series reprogramming baseline",
            "modern_family": "LLM for time series",
            "fit_to_our_task": "low_medium",
            "why_it_fits": "It is visibly modern and can serve as an external check if mature tooling is available.",
            "data_support": "low",
            "implementation_effort": "high",
            "a800_usefulness": "high_but_costly",
            "expected_value": "low_medium",
            "main_risk": "Likely expensive, hard to use with exogenous variables, and may distract from the paper's market-risk problem.",
            "paper_role_if_successful": "appendix only",
            "decision": "do_not_prioritize",
        },
    ]
    return pd.DataFrame(rows)


def action_plan() -> pd.DataFrame:
    rows = [
        {
            "step": "C58a",
            "action": "Check local/remote feasibility for mamba_ssm or a practical selective-SSM implementation on A800.",
            "success_criterion": "Can train a rolling-month supervised sequence tail-risk model with exogenous features and output tail probability/quantile scores.",
            "if_fail": "Do not force Mamba name; move to foundation-model baseline check.",
        },
        {
            "step": "C58b",
            "action": "Check feasibility of Chronos/TimesFM/Moirai zero-shot or light fine-tune baseline.",
            "success_criterion": "Can produce calibrated quantiles/scenarios for spread and enter fixed-budget evaluation without leakage.",
            "if_fail": "Keep C31 as credible modern baseline and report C32/C33 honestly as representative diagnostics.",
        },
        {
            "step": "C59",
            "action": "Implement the selected stronger baseline under the frozen C57 metric protocol.",
            "success_criterion": "Outputs risk scores for all evaluated months; enters C54-style top30/multi-budget robustness.",
            "if_fail": "Record negative result; do not let it contaminate the main claim.",
        },
        {
            "step": "C60",
            "action": "Compare stronger baseline against C44/C50 and update paper claim boundary.",
            "success_criterion": "If it improves top30 or multi-budget robustness, promote appropriately; otherwise use as hard baseline defense.",
            "if_fail": "No promotion; keep current C44/C50 bounded narrative.",
        },
    ]
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    feasibility = feasibility_table()
    plan = action_plan()

    feasibility.to_csv(OUT / "c58_modern_baseline_feasibility_matrix.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUT / "c58_minimal_implementation_plan.csv", index=False, encoding="utf-8-sig")

    report = [
        "# C58 Stronger Modern Baseline Feasibility",
        "",
        "Purpose: decide whether and how to add one stronger modern baseline without changing the paper's problem definition or creating a stitched model.",
        "",
        "## Feasibility Matrix",
        "",
        md_table(
            feasibility,
            [
                "candidate",
                "modern_family",
                "fit_to_our_task",
                "data_support",
                "implementation_effort",
                "expected_value",
                "decision",
            ],
            10,
        ),
        "",
        "## Minimal Implementation Plan",
        "",
        md_table(plan, ["step", "action", "success_criterion", "if_fail"], 10),
        "",
        "## Decision",
        "",
        "The next practical move should not be KAN or a generic LLM. The first feasibility check should be a supervised Mamba/S4-style sequence baseline, because it can use our exogenous market/mechanism features and still produce tail-risk scores under the frozen fixed-budget protocol.",
        "",
        "If true Mamba/S4 dependencies are impractical, the second-best hard baseline is a time-series foundation model such as Chronos/TimesFM/Moirai, used as an external modern forecasting baseline. It should not replace the mechanism-aware method unless it improves fixed-budget risk coverage.",
        "",
        "A diffusion-transformer baseline is scientifically aligned with probabilistic forecasting, but a clean implementation may be too costly for the current stage; it is deferred unless the first two options fail or remain too weak.",
    ]
    (OUT / "c58_stronger_modern_baseline_feasibility_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
