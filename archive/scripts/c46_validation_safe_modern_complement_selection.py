from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import BUDGETS, build_alerts, eval_alert, prepare_scores


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "c38_modern_probabilistic_complement_audit" / "c38_modern_complement_predictions.csv"
OUT = ROOT / "c46_validation_safe_modern_complement_selection"
OUT.mkdir(exist_ok=True)

REFERENCE = "C44_limited_budget_band_until30"
BASE = "C20b_base"

CANDIDATE_SCORES = [
    "c22_deep_w20",
    "protected_core_alert_score_proxy",
    "c22_modern_w10",
    "c22_diffusion_w10",
    "c22_deep_w10",
    "c22_modern_w20",
    "c22_diffusion_w20",
    "c22_deep_w30",
    "c22_modern_w30",
    "c22_diffusion_w30",
    "base_modern_w10",
    "base_modern_w20",
    "c31_blend_c31_tail_prob_iso_base_w0p2",
    "c31_blend_c31_conf_q20_risk_base_w0p2",
    "c33_blend_c33_diff_q20_risk_base_w0p2",
    "c32_blend_c32_ssm_tail_prob_iso_base_w0p2",
]


def monthly_alert_mask(df: pd.DataFrame, score_col: str, budget: float) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, mdf in df.groupby("test_month", sort=True):
        k = max(1, int(np.ceil(len(mdf) * budget)))
        chosen = mdf.sort_values(score_col, ascending=False).head(k).index
        alert.loc[chosen] = True
    return alert


def build_candidate_alerts(df: pd.DataFrame, budget: float) -> dict[str, pd.Series]:
    c44_df = prepare_scores(df)
    c44_alerts = build_alerts(c44_df, budget)
    alerts = {
        BASE: c44_alerts[BASE],
        REFERENCE: c44_alerts["limited_budget_band_until30"],
    }
    for col in CANDIDATE_SCORES:
        if col in df.columns:
            alerts[col] = monthly_alert_mask(df, col, budget)
    return alerts


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "validation_non_pressure": df[~df["test_month"].isin(PRESSURE_MONTHS)],
        "holdout_pressure": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "all_available_months": df,
    }
    rows = []
    for budget in BUDGETS:
        alerts = build_candidate_alerts(df, budget)
        for method, alert in alerts.items():
            for scope, sdf in scopes.items():
                if sdf.empty:
                    continue
                met = eval_alert(sdf, alert.loc[sdf.index])
                met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
                rows.append({"budget": budget, "method": method, "scope": scope, **met})
    return pd.DataFrame(rows)


def delta_to_reference(metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    ref = metrics[metrics["method"].eq(reference)][
        ["budget", "scope", "recall", "excess_share", "false_alert_burden", "gate_score"]
    ].rename(
        columns={
            "recall": "ref_recall",
            "excess_share": "ref_excess_share",
            "false_alert_burden": "ref_false_alert_burden",
            "gate_score": "ref_gate_score",
        }
    )
    out = metrics.merge(ref, on=["budget", "scope"], how="left")
    out["delta_recall"] = out["recall"] - out["ref_recall"]
    out["delta_excess_share"] = out["excess_share"] - out["ref_excess_share"]
    out["delta_false_alert_burden"] = out["false_alert_burden"] - out["ref_false_alert_burden"]
    out["delta_gate_score"] = out["gate_score"] - out["ref_gate_score"]
    return out


def validation_select(delta_ref: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    val = delta_ref[
        delta_ref["scope"].eq("validation_non_pressure")
        & ~delta_ref["method"].isin([BASE, REFERENCE])
        & delta_ref["budget"].isin([0.25, 0.30, 0.35, 0.40])
    ].copy()
    rows = []
    for method, g in val.groupby("method"):
        tight = g[g["budget"].isin([0.25, 0.30])]
        broad = g[g["budget"].isin([0.35, 0.40])]
        rows.append(
            {
                "method": method,
                "val_tight_mean_gate_delta_vs_c44": tight["delta_gate_score"].mean(),
                "val_tight_min_gate_delta_vs_c44": tight["delta_gate_score"].min(),
                "val_tight_min_recall_delta_vs_c44": tight["delta_recall"].min(),
                "val_tight_min_excess_delta_vs_c44": tight["delta_excess_share"].min(),
                "val_broad_min_gate_delta_vs_c44": broad["delta_gate_score"].min(),
                "val_broad_min_recall_delta_vs_c44": broad["delta_recall"].min(),
                "val_broad_min_excess_delta_vs_c44": broad["delta_excess_share"].min(),
            }
        )
    score = pd.DataFrame(rows)
    safe = score[
        (score["val_tight_min_gate_delta_vs_c44"] >= -1e-12)
        & (score["val_tight_min_recall_delta_vs_c44"] >= -1e-12)
        & (score["val_tight_min_excess_delta_vs_c44"] >= -1e-12)
        & (score["val_broad_min_gate_delta_vs_c44"] >= -1e-12)
        & (score["val_broad_min_recall_delta_vs_c44"] >= -1e-12)
        & (score["val_broad_min_excess_delta_vs_c44"] >= -1e-12)
    ].copy()
    if safe.empty:
        selected = REFERENCE
        score["selected_by_strict_validation_rule"] = False
        score["strict_reason"] = "no_modern_candidate_dominates_c44_on_validation"
        return selected, score.sort_values("val_tight_mean_gate_delta_vs_c44", ascending=False)
    safe = safe.sort_values(
        [
            "val_tight_mean_gate_delta_vs_c44",
            "val_tight_min_gate_delta_vs_c44",
            "val_tight_min_excess_delta_vs_c44",
        ],
        ascending=False,
    )
    selected = str(safe.iloc[0]["method"])
    score["selected_by_strict_validation_rule"] = score["method"].eq(selected)
    score["strict_reason"] = "candidate_dominates_c44_on_validation"
    return selected, score.sort_values(
        ["selected_by_strict_validation_rule", "val_tight_mean_gate_delta_vs_c44"], ascending=[False, False]
    )


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for v in row:
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    metrics = evaluate(df)
    delta_c44 = delta_to_reference(metrics, REFERENCE)
    delta_base = delta_to_reference(metrics, BASE)
    selected, selection = validation_select(delta_c44)

    metrics.to_csv(OUT / "c46_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    delta_c44.to_csv(OUT / "c46_delta_vs_c44.csv", index=False, encoding="utf-8-sig")
    delta_base.to_csv(OUT / "c46_delta_vs_base.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "c46_validation_selection.csv", index=False, encoding="utf-8-sig")

    pressure30 = metrics[metrics["scope"].eq("holdout_pressure") & metrics["budget"].eq(0.30)].copy()
    pressure30 = pressure30.sort_values(["gate_score", "recall", "excess_share"], ascending=False)
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].copy()
    all30 = all30.sort_values(["gate_score", "recall", "excess_share"], ascending=False)
    selected_delta = delta_c44[delta_c44["method"].eq(selected)].sort_values(["scope", "budget"])

    report = [
        "# C46 Validation-Safe Modern Complement Selection",
        "",
        f"Reference validation-frozen rule: `{REFERENCE}`.",
        "",
        "Selection rule: a modern candidate may replace C44 only if it is no worse than C44 on validation tight budgets and broad budgets for recall, excess_share, and gate_score.",
        "",
        f"Selected by strict validation rule: `{selected}`.",
        "",
        "## Validation Selection",
        "",
        md_table(selection, list(selection.columns), 25),
        "",
        "## Holdout Pressure 30% Ranking",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## All-Month 30% Ranking",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## Selected Delta vs C44",
        "",
        md_table(
            selected_delta,
            ["scope", "budget", "recall", "excess_share", "false_alert_burden", "gate_score", "delta_recall", "delta_excess_share", "delta_gate_score"],
            20,
        ),
        "",
        "## Interpretation",
        "",
        "- If no modern candidate dominates C44 on validation, C44 remains the main validation-frozen rule.",
        "- Modern candidates that perform well on holdout but fail validation dominance are strong baselines, not main claims.",
        "- This avoids converting a promising modern complement into another test-set-selected hybrid.",
    ]
    (OUT / "c46_validation_safe_modern_complement_selection_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
