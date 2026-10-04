from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import build_alerts, eval_alert, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED, monthly_alert_mask
from c49_pressure_transfer_safe_candidate_selection import candidate_alert


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "taskB_main_result_strengthening" / "c77_stress_gated_deep_boundary_enhancement"
OUT.mkdir(parents=True, exist_ok=True)

BUDGET = 0.30
C44 = "C44_limited_budget_band_until30"
C50 = "c22_deep_w20_until30_else_c44"
DEEP_SCORE = "c22_deep_w20"
STRESS_FEATURES = [
    "da_cong_node_abs_mean",
    "da_cong_node_abs_max",
    "da_cong_node_abs_mean_delta48",
    "net_load_pred",
    "renewable_share",
    "neg_spread_lag_48",
    "hist_tail65_rate_7d",
]


def metric(df: pd.DataFrame, alert: pd.Series) -> dict:
    met = eval_alert(df, alert.loc[df.index])
    met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
    return met


def scopes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "validation_non_pressure": df[~df["test_month"].isin(PRESSURE_MONTHS)],
        "holdout_pressure": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "all_available_months": df,
    }


def threshold_from_validation(df: pd.DataFrame, feature: str, q: float) -> float:
    val = df[~df["test_month"].isin(PRESSURE_MONTHS)]
    return float(pd.to_numeric(val[feature], errors="coerce").quantile(q))


def stress_gated_alert(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    replace_frac: float,
    candidate_pool_pct: float,
) -> pd.Series:
    """Replace a small C44 boundary slice with ex-ante stress + deep-score samples.

    The rule is intentionally conservative:
    - start from the C44 alert set;
    - protect the top base-risk core;
    - only replace the lowest-ranked C44 boundary samples;
    - only add samples whose ex-ante stress feature exceeds a validation-derived threshold.
    """
    scored = prepare_scores(df.copy())
    c44 = build_alerts(scored, BUDGET)["limited_budget_band_until30"].copy()
    alert = c44.copy()
    feature_values = pd.to_numeric(scored[feature], errors="coerce")
    stress = feature_values >= threshold
    if DEEP_SCORE not in scored.columns:
        return alert
    for _, mdf in scored.groupby("test_month", sort=True):
        k = max(1, int(np.ceil(len(mdf) * BUDGET)))
        n_replace = max(1, int(np.floor(k * replace_frac)))
        month_idx = mdf.index
        current = alert.loc[month_idx]
        if int(current.sum()) <= n_replace:
            continue
        protected_core_cut = BUDGET - 0.075
        current_df = mdf[current.loc[month_idx].to_numpy()].copy()
        current_df = current_df[current_df["base_rank_pct"] > protected_core_cut]
        if current_df.empty:
            continue
        drop_idx = list(current_df.sort_values("base_score", ascending=True).head(n_replace).index)
        candidates = mdf[~current.loc[month_idx].to_numpy()].copy()
        candidates = candidates[candidates["base_rank_pct"] <= candidate_pool_pct]
        candidates = candidates[stress.loc[candidates.index]]
        if candidates.empty:
            continue
        add_idx = list(candidates.sort_values(DEEP_SCORE, ascending=False).head(len(drop_idx)).index)
        if not add_idx:
            continue
        alert.loc[drop_idx[: len(add_idx)]] = False
        alert.loc[add_idx] = True
    return alert


def evaluate_alerts(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for scope, sdf in scopes(df).items():
        for method, alert in alerts.items():
            rows.append({"scope": scope, "budget": BUDGET, "method": method, **metric(sdf, alert)})
    out = pd.DataFrame(rows)
    ref = out[out["method"].eq(C44)][["scope", "recall", "excess_share", "gate_score"]].rename(
        columns={"recall": "c44_recall", "excess_share": "c44_excess_share", "gate_score": "c44_gate_score"}
    )
    base = out[out["method"].eq("C20b_base")][["scope", "recall", "excess_share", "gate_score"]].rename(
        columns={"recall": "base_recall", "excess_share": "base_excess_share", "gate_score": "base_gate_score"}
    )
    out = out.merge(ref, on="scope", how="left").merge(base, on="scope", how="left")
    out["delta_recall_vs_c44"] = out["recall"] - out["c44_recall"]
    out["delta_excess_vs_c44"] = out["excess_share"] - out["c44_excess_share"]
    out["delta_gate_vs_c44"] = out["gate_score"] - out["c44_gate_score"]
    out["delta_gate_vs_base"] = out["gate_score"] - out["base_gate_score"]
    return out


def grid_search(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = prepare_scores(df.copy())
    base_alerts = build_alerts(scored, BUDGET)
    alerts_base = {
        "C20b_base": base_alerts["C20b_base"],
        C44: base_alerts["limited_budget_band_until30"],
        "C50_deep_w20_until30_else_c44": candidate_alert(scored, C50, BUDGET),
        "Deep_direct_c22_w20": monthly_alert_mask(scored, DEEP_SCORE, BUDGET),
    }
    rows = []
    candidate_alerts: dict[str, pd.Series] = {}
    for feature in STRESS_FEATURES:
        if feature not in scored.columns:
            continue
        for q in [0.60, 0.70, 0.80]:
            threshold = threshold_from_validation(scored, feature, q)
            for replace_frac in [0.05, 0.10, 0.15]:
                for pool in [0.40, 0.50]:
                    name = f"C77_{feature}_q{int(q*100)}_r{int(replace_frac*100)}_pool{int(pool*100)}"
                    candidate_alerts[name] = stress_gated_alert(scored, feature, threshold, replace_frac, pool)
    metrics = evaluate_alerts(scored, {**alerts_base, **candidate_alerts})

    for method, g in metrics.groupby("method", sort=True):
        if not method.startswith("C77_"):
            continue
        val = g[g["scope"].eq("validation_non_pressure")].iloc[0]
        hold = g[g["scope"].eq("holdout_pressure")].iloc[0]
        allm = g[g["scope"].eq("all_available_months")].iloc[0]
        rows.append(
            {
                "method": method,
                "val_delta_gate_vs_c44": val["delta_gate_vs_c44"],
                "val_delta_recall_vs_c44": val["delta_recall_vs_c44"],
                "val_delta_excess_vs_c44": val["delta_excess_vs_c44"],
                "holdout_delta_gate_vs_c44": hold["delta_gate_vs_c44"],
                "holdout_delta_recall_vs_c44": hold["delta_recall_vs_c44"],
                "holdout_delta_excess_vs_c44": hold["delta_excess_vs_c44"],
                "all_delta_gate_vs_c44": allm["delta_gate_vs_c44"],
                "all_delta_recall_vs_c44": allm["delta_recall_vs_c44"],
                "all_delta_excess_vs_c44": allm["delta_excess_vs_c44"],
            }
        )
    screen = pd.DataFrame(rows)
    if screen.empty:
        return metrics, screen
    screen["passes_validation_gate"] = (
        (screen["val_delta_gate_vs_c44"] > 0)
        & (screen["val_delta_recall_vs_c44"] >= 0)
        & (screen["val_delta_excess_vs_c44"] >= 0)
    )
    screen["passes_holdout_safety"] = (
        (screen["holdout_delta_gate_vs_c44"] >= 0)
        & (screen["holdout_delta_recall_vs_c44"] >= -1e-12)
        & (screen["holdout_delta_excess_vs_c44"] >= 0)
    )
    screen["passes_all_month_safety"] = screen["all_delta_gate_vs_c44"] >= 0
    screen["promotable"] = (
        screen["passes_validation_gate"] & screen["passes_holdout_safety"] & screen["passes_all_month_safety"]
    )
    screen = screen.sort_values(
        ["promotable", "holdout_delta_gate_vs_c44", "all_delta_gate_vs_c44", "val_delta_gate_vs_c44"],
        ascending=[False, False, False, False],
    )
    return metrics, screen


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    metrics, screen = grid_search(df)
    metrics.to_csv(OUT / "c77_all_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    screen.to_csv(OUT / "c77_candidate_screen.csv", index=False, encoding="utf-8-sig")

    promoted = screen[screen["promotable"].eq(True)] if not screen.empty else pd.DataFrame()
    selected_name = None if promoted.empty else str(promoted.iloc[0]["method"])
    if selected_name:
        selected_metrics = metrics[metrics["method"].eq(selected_name)].copy()
    else:
        selected_metrics = pd.DataFrame()
    selected_metrics.to_csv(OUT / "c77_selected_metrics.csv", index=False, encoding="utf-8-sig")

    decision = pd.DataFrame(
        [
            {
                "decision": "PROMOTE_C77" if selected_name else "NO_GO_C77",
                "selected_method": selected_name if selected_name else "",
                "reason": (
                    "A stress-gated deep boundary candidate passed validation, holdout pressure, and all-month safety gates."
                    if selected_name
                    else "No stress-gated deep boundary candidate passed validation, holdout pressure, and all-month safety gates."
                ),
            }
        ]
    )
    decision.to_csv(OUT / "c77_decision.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C77 Stress-Gated Deep Boundary Enhancement",
        "",
        "Purpose: test whether ex-ante stress-gated use of the deep signal can improve C44/C50 without month-specific repair.",
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False),
        "",
        "## Top Candidate Screen",
        "",
        screen.head(20).to_markdown(index=False) if not screen.empty else "No candidates generated.",
        "",
        "## Selected Metrics",
        "",
        selected_metrics.to_markdown(index=False) if not selected_metrics.empty else "No selected C77 candidate.",
        "",
    ]
    (OUT / "c77_stress_gated_deep_boundary_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
