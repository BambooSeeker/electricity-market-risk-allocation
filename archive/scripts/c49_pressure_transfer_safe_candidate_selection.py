from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import BUDGETS, build_alerts, eval_alert, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED, monthly_alert_mask


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c49_pressure_transfer_safe_candidate_selection"
OUT.mkdir(exist_ok=True)

REF = "C44_limited_budget_band_until30"
CANDIDATES = [
    "protected_core_alert_score_proxy_until30_else_c44",
    "c22_deep_w20_until30_else_c44",
    "c22_modern_w10_until30_else_c44",
    "c22_modern_w20_until30_else_c44",
    "c22_modern_w30_until30_else_c44",
    "c22_deep_w30_until30_else_c44",
    "c31_blend_c31_tail_prob_iso_base_w0p2_until30_else_c44",
]
RAW_SCORE = {
    "protected_core_alert_score_proxy_until30_else_c44": "protected_core_alert_score_proxy",
    "c22_deep_w20_until30_else_c44": "c22_deep_w20",
    "c22_modern_w10_until30_else_c44": "c22_modern_w10",
    "c22_modern_w20_until30_else_c44": "c22_modern_w20",
    "c22_modern_w30_until30_else_c44": "c22_modern_w30",
    "c22_deep_w30_until30_else_c44": "c22_deep_w30",
    "c31_blend_c31_tail_prob_iso_base_w0p2_until30_else_c44": "c31_blend_c31_tail_prob_iso_base_w0p2",
}


def c44_alert(df: pd.DataFrame, budget: float) -> pd.Series:
    return build_alerts(prepare_scores(df.copy()), budget)["limited_budget_band_until30"]


def candidate_alert(df: pd.DataFrame, method: str, budget: float) -> pd.Series:
    ref = c44_alert(df, budget)
    if method == REF:
        return ref
    raw = monthly_alert_mask(df, RAW_SCORE[method], budget)
    return raw if budget <= 0.30 else ref


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "validation_non_pressure": df[~df["test_month"].isin(PRESSURE_MONTHS)],
        "holdout_pressure": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "all_available_months": df,
    }
    rows = []
    for budget in BUDGETS:
        alerts = {REF: candidate_alert(df, REF, budget)}
        for method in CANDIDATES:
            alerts[method] = candidate_alert(df, method, budget)
        for method, alert in alerts.items():
            for scope, sdf in scopes.items():
                met = eval_alert(sdf, alert.loc[sdf.index])
                met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
                rows.append({"budget": budget, "method": method, "scope": scope, **met})
    return pd.DataFrame(rows)


def delta_vs_ref(metrics: pd.DataFrame) -> pd.DataFrame:
    ref = metrics[metrics["method"].eq(REF)][
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
    out["delta_gate_score"] = out["gate_score"] - out["ref_gate_score"]
    return out


def displacement_profile(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_scores(df.copy())
    rows = []
    scopes = {
        "validation_non_pressure": df[~df["test_month"].isin(PRESSURE_MONTHS)],
        "holdout_pressure": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "all_available_months": df,
    }
    for budget in [0.25, 0.30]:
        ref_alert = c44_alert(df, budget)
        for method in CANDIDATES:
            cand = candidate_alert(df, method, budget)
            for scope, sdf in scopes.items():
                added = cand.loc[sdf.index] & ~ref_alert.loc[sdf.index]
                dropped = ref_alert.loc[sdf.index] & ~cand.loc[sdf.index]
                for group_name, mask in {"added": added, "dropped": dropped}.items():
                    g = sdf[mask]
                    rows.append(
                        {
                            "budget": budget,
                            "method": method,
                            "scope": scope,
                            "group": group_name,
                            "n": len(g),
                            "tail_events": int(g["negative_tail"].sum()) if len(g) else 0,
                            "tail_event_rate": float(g["negative_tail"].mean()) if len(g) else np.nan,
                            "negative_excess_sum": float(g["negative_excess"].sum()) if len(g) else 0.0,
                            "boundary_bonus_mean": float(g["boundary_bonus"].mean()) if len(g) else np.nan,
                            "boundary_bonus_p75": float(g["boundary_bonus"].quantile(0.75)) if len(g) else np.nan,
                            "high_boundary_rate": float((g["boundary_bonus"] >= 0.70).mean()) if len(g) else np.nan,
                        }
                    )
    prof = pd.DataFrame(rows)
    wide = prof.pivot_table(
        index=["budget", "method", "scope"],
        columns="group",
        values=["n", "tail_events", "tail_event_rate", "negative_excess_sum", "boundary_bonus_mean", "high_boundary_rate"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    wide = wide.reset_index()
    wide["boundary_displacement_gap"] = wide["boundary_bonus_mean_dropped"] - wide["boundary_bonus_mean_added"]
    wide["high_boundary_displacement_gap"] = wide["high_boundary_rate_dropped"] - wide["high_boundary_rate_added"]
    wide["tail_event_rate_gap_holdout_only"] = wide["tail_event_rate_dropped"] - wide["tail_event_rate_added"]
    return prof, wide


def select_candidate(delta: pd.DataFrame, disp_wide: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    val = delta[
        delta["scope"].eq("validation_non_pressure")
        & delta["method"].isin(CANDIDATES)
        & delta["budget"].isin([0.25, 0.30, 0.35, 0.40])
    ].copy()
    disp_val = disp_wide[
        disp_wide["scope"].eq("validation_non_pressure")
        & disp_wide["method"].isin(CANDIDATES)
        & disp_wide["budget"].isin([0.25, 0.30])
    ].copy()
    rows = []
    for method, g in val.groupby("method"):
        tight = g[g["budget"].isin([0.25, 0.30])]
        broad = g[g["budget"].isin([0.35, 0.40])]
        d = disp_val[disp_val["method"].eq(method)]
        rows.append(
            {
                "method": method,
                "val_tight_mean_gate_delta": tight["delta_gate_score"].mean(),
                "val_tight_min_gate_delta": tight["delta_gate_score"].min(),
                "val_tight_min_recall_delta": tight["delta_recall"].min(),
                "val_tight_min_excess_delta": tight["delta_excess_share"].min(),
                "val_broad_min_gate_delta": broad["delta_gate_score"].min(),
                "val_broad_min_recall_delta": broad["delta_recall"].min(),
                "val_broad_min_excess_delta": broad["delta_excess_share"].min(),
                "max_boundary_displacement_gap": d["boundary_displacement_gap"].max(),
                "max_high_boundary_displacement_gap": d["high_boundary_displacement_gap"].max(),
            }
        )
    score = pd.DataFrame(rows)

    # Pressure-transfer-safe rule: besides validation improvement, do not choose
    # candidates that strongly replace C44's high-boundary samples. Thresholds are
    # feature-only and evaluated on validation/non-pressure data.
    safe = score[
        (score["val_tight_min_gate_delta"] >= -1e-12)
        & (score["val_tight_min_recall_delta"] >= -1e-12)
        & (score["val_tight_min_excess_delta"] >= -1e-12)
        & (score["val_broad_min_gate_delta"] >= -1e-12)
        & (score["val_broad_min_recall_delta"] >= -1e-12)
        & (score["val_broad_min_excess_delta"] >= -1e-12)
        & (score["max_boundary_displacement_gap"] <= 0.25)
        & (score["max_high_boundary_displacement_gap"] <= 0.20)
    ].copy()
    if safe.empty:
        selected = REF
        score["selected_by_pressure_transfer_safe_rule"] = False
        score["reason"] = "no_candidate_satisfies_validation_and_boundary_protection"
        return selected, score.sort_values(["val_tight_mean_gate_delta"], ascending=False)
    safe = safe.sort_values(
        ["val_tight_mean_gate_delta", "val_tight_min_gate_delta", "max_boundary_displacement_gap"],
        ascending=[False, False, True],
    )
    selected = str(safe.iloc[0]["method"])
    score["selected_by_pressure_transfer_safe_rule"] = score["method"].eq(selected)
    score["reason"] = "validation_gain_with_boundary_displacement_control"
    return selected, score.sort_values(
        ["selected_by_pressure_transfer_safe_rule", "val_tight_mean_gate_delta"], ascending=[False, False]
    )


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    metrics = evaluate(df)
    delta = delta_vs_ref(metrics)
    disp_long, disp_wide = displacement_profile(df)
    selected, selection = select_candidate(delta, disp_wide)

    metrics.to_csv(OUT / "c49_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "c49_delta_vs_c44.csv", index=False, encoding="utf-8-sig")
    disp_long.to_csv(OUT / "c49_displacement_long.csv", index=False, encoding="utf-8-sig")
    disp_wide.to_csv(OUT / "c49_displacement_wide.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "c49_pressure_transfer_safe_selection.csv", index=False, encoding="utf-8-sig")

    selected_delta = delta[delta["method"].eq(selected)].sort_values(["scope", "budget"])
    holdout30 = delta[delta["scope"].eq("holdout_pressure") & delta["budget"].eq(0.30)].sort_values(
        ["delta_gate_score", "delta_recall", "delta_excess_share"], ascending=False
    )
    val_disp = disp_wide[disp_wide["scope"].eq("validation_non_pressure") & disp_wide["budget"].eq(0.30)].sort_values(
        ["boundary_displacement_gap", "high_boundary_displacement_gap"], ascending=True
    )
    holdout_disp = disp_wide[disp_wide["scope"].eq("holdout_pressure") & disp_wide["budget"].eq(0.30)].sort_values(
        ["tail_event_rate_gap_holdout_only"], ascending=False
    )

    report = [
        "# C49 Pressure-Transfer-Safe Candidate Selection",
        "",
        "Purpose: choose modern complements using validation performance plus feature-only boundary-displacement control, rather than validation mean alone.",
        "",
        f"Selected method: `{selected}`.",
        "",
        "## Selection Table",
        "",
        md_table(selection, list(selection.columns), 30),
        "",
        "## Selected Delta vs C44",
        "",
        md_table(
            selected_delta,
            ["scope", "budget", "recall", "excess_share", "false_alert_burden", "gate_score", "delta_recall", "delta_excess_share", "delta_gate_score"],
            20,
        ),
        "",
        "## Holdout Pressure Top30: Candidate Delta vs C44",
        "",
        md_table(
            holdout30,
            ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "delta_recall", "delta_excess_share", "delta_gate_score"],
            25,
        ),
        "",
        "## Validation Top30 Boundary Displacement",
        "",
        md_table(
            val_disp,
            ["method", "boundary_displacement_gap", "high_boundary_displacement_gap", "boundary_bonus_mean_added", "boundary_bonus_mean_dropped", "high_boundary_rate_added", "high_boundary_rate_dropped"],
            25,
        ),
        "",
        "## Holdout Top30 Tail Displacement Diagnostic",
        "",
        md_table(
            holdout_disp,
            ["method", "tail_event_rate_gap_holdout_only", "tail_event_rate_added", "tail_event_rate_dropped", "negative_excess_sum_added", "negative_excess_sum_dropped", "boundary_displacement_gap"],
            25,
        ),
        "",
        "## Interpretation",
        "",
        "- The selection rule uses no pressure-month labels; pressure months are reported only as holdout evidence.",
        "- Boundary-displacement control rejects candidates that replace too many high-boundary C44 samples, the failure mechanism identified in C47.",
        "- If the selected method still fails holdout pressure, modern complements should remain baselines rather than main method components.",
    ]
    (OUT / "c49_pressure_transfer_safe_candidate_selection_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
