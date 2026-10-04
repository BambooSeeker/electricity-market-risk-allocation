from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import build_alerts, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED
from c49_pressure_transfer_safe_candidate_selection import candidate_alert
from c77_stress_gated_deep_boundary_enhancement import C50, stress_gated_alert, threshold_from_validation


OUT = ROOT / "manuscript_ready_artifacts" / "case_analysis_upgrade"
OUT.mkdir(parents=True, exist_ok=True)

BUDGET = 0.30
STRESS_FEATURE = "net_load_pred"
STRESS_Q = 0.60
REPLACE_FRAC = 0.05
CANDIDATE_POOL = 0.50


def summarize_group(df: pd.DataFrame, idx, month: str, comparison: str, group: str) -> dict:
    g = df.loc[list(idx)].copy()
    if g.empty:
        return {
            "month": month,
            "comparison": comparison,
            "group": group,
            "changed_intervals": 0,
            "tail_events": 0,
            "tail_event_rate": 0.0,
            "negative_excess_sum": 0.0,
            "negative_excess_mean": 0.0,
            "stress_feature_mean": 0.0,
            "deep_score_mean": 0.0,
            "base_score_mean": 0.0,
            "base_rank_pct_mean": 0.0,
            "da_congestion_abs_mean": 0.0,
        }
    da_cols = [c for c in g.columns if c.startswith("price_day_ahead_cong")]
    da_abs = g[da_cols].abs().mean(axis=1) if da_cols else pd.Series(0.0, index=g.index)
    return {
        "month": month,
        "comparison": comparison,
        "group": group,
        "changed_intervals": int(len(g)),
        "tail_events": int(g["negative_tail"].sum()),
        "tail_event_rate": float(g["negative_tail"].mean()),
        "negative_excess_sum": float(g["negative_excess"].sum()),
        "negative_excess_mean": float(g["negative_excess"].mean()),
        "stress_feature_mean": float(g[STRESS_FEATURE].mean()),
        "deep_score_mean": float(g["c22_deep_w20"].mean()) if "c22_deep_w20" in g else 0.0,
        "base_score_mean": float(g["base_score"].mean()) if "base_score" in g else 0.0,
        "base_rank_pct_mean": float(g["base_rank_pct"].mean()) if "base_rank_pct" in g else 0.0,
        "da_congestion_abs_mean": float(da_abs.mean()),
    }


def write_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    scored = prepare_scores(df)
    base_alerts = build_alerts(scored, BUDGET)
    boundary = base_alerts["limited_budget_band_until30"]
    ungated = candidate_alert(scored, C50, BUDGET)
    threshold = threshold_from_validation(scored, STRESS_FEATURE, STRESS_Q)
    stress_gated = stress_gated_alert(
        scored,
        feature=STRESS_FEATURE,
        threshold=threshold,
        replace_frac=REPLACE_FRAC,
        candidate_pool_pct=CANDIDATE_POOL,
    )

    rows = []
    comparisons = [
        ("stress_gated_vs_boundary", stress_gated, boundary),
        ("stress_gated_vs_ungated", stress_gated, ungated),
    ]
    for month, mdf in scored[scored["test_month"].isin(PRESSURE_MONTHS)].groupby("test_month", sort=True):
        midx = mdf.index
        for name, method_alert, ref_alert in comparisons:
            method_set = set(midx[method_alert.loc[midx].to_numpy()])
            ref_set = set(midx[ref_alert.loc[midx].to_numpy()])
            added = method_set - ref_set
            dropped = ref_set - method_set
            rows.append(summarize_group(scored, added, str(month), name, "added_by_stress_gated"))
            rows.append(summarize_group(scored, dropped, str(month), name, "dropped_by_stress_gated"))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stress_gated_replacement_mechanism_pressure.csv", index=False, encoding="utf-8-sig")

    display = out.copy()
    for c in [
        "tail_event_rate",
        "negative_excess_sum",
        "negative_excess_mean",
        "stress_feature_mean",
        "deep_score_mean",
        "base_score_mean",
        "base_rank_pct_mean",
        "da_congestion_abs_mean",
    ]:
        display[c] = display[c].round(4)
    write_md(display, OUT / "stress_gated_replacement_mechanism_pressure.md")


if __name__ == "__main__":
    main()
