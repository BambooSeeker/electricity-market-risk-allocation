from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS, rank01
from c6_multinode_stageb_risk_model import load_multinode_raw


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "c8_vmd_enhanced_sequence_diagnostic" / "c8_vmd_sequence_predictions.csv"
OUT = ROOT / "c9_pressure_miss_diagnosis_and_conditioned_adjustment"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
BASE_SCORES = [
    "AllSignals_equal_rank_avg",
    "HGB_q10_distance_two_stage",
    "QIA_two_stage",
    "tcn_s48_h64_w6_vmd_hgb_blend",
    "tcn_s48_h64_w6_vmd_all_blend",
    "tcn_s48_h64_w6_vmd_qia_blend",
]


def metric_from_mask(df: pd.DataFrame, alert: np.ndarray) -> dict[str, float]:
    y = df["negative_tail"].astype(int).to_numpy()
    excess = df["negative_excess"].astype(float).to_numpy()
    hit = y[alert].sum()
    events = y.sum()
    alerts = alert.sum()
    precision = hit / alerts if alerts else np.nan
    recall = hit / events if events else np.nan
    alert_excess = excess[alert].sum()
    total_excess = excess.sum()
    return {
        "n": len(df),
        "events": int(events),
        "alerts": int(alerts),
        "precision": float(precision),
        "recall": float(recall),
        "excess_share": float(alert_excess / max(total_excess, 1e-9)),
        "missed_event_share": float(1 - recall),
        "missed_excess_share": float(1 - alert_excess / max(total_excess, 1e-9)),
        "false_alert_burden": float(1 - precision),
    }


def monthly_alert_mask(df: pd.DataFrame, score_col: str, budget: float) -> np.ndarray:
    masks = []
    for _, mdf in df.groupby("test_month", sort=True):
        score = mdf[score_col].to_numpy()
        order = np.argsort(-score)
        mask = np.zeros(len(mdf), dtype=bool)
        mask[order[: max(1, int(np.ceil(len(mdf) * budget)))]] = True
        masks.append(mask)
    return np.concatenate(masks)


def summarize(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    rows = []
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for score in score_cols:
            for budget in BUDGETS:
                alert = monthly_alert_mask(sdf, score, budget)
                met = metric_from_mask(sdf, alert)
                rows.append({"scope": scope, "method": score, "budget": budget, **met})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def add_history_features(raw: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    df = raw.sort_values("time").copy()
    df["spread_rt_minus_da"] = target
    s = df["spread_rt_minus_da"]
    for lag in [48, 96, 336]:
        df[f"spread_lag_{lag}"] = s.shift(lag)
    df["hist_tail65_rate_7d"] = (s.shift(48) < -65).rolling(336, min_periods=48).mean()
    df["hist_tail100_rate_7d"] = (s.shift(48) < -100).rolling(336, min_periods=48).mean()
    df["spread_roll_7d_min"] = s.shift(48).rolling(336, min_periods=48).min()
    return df.drop(columns=["spread_rt_minus_da"])


def load_target_series() -> pd.DataFrame:
    # Reuse the target embedded in C8 predictions for evaluated months; raw history is only used for lag construction.
    pred = pd.read_csv(PRED, parse_dates=["time"], usecols=["time", "spread_rt_minus_da"])
    return pred.sort_values("time").drop_duplicates("time", keep="last")


def build_panel() -> pd.DataFrame:
    pred = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    raw = load_multinode_raw()
    target = load_target_series()
    raw = raw.merge(target, on="time", how="left")
    raw = add_history_features(raw, raw["spread_rt_minus_da"])
    df = pred.merge(raw, on="time", how="left")
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour"] = df["time"].dt.hour
    df["is_midday"] = ((hour >= 11) & (hour <= 15.5)).astype(float)
    df["is_morning_ramp"] = ((hour >= 6) & (hour <= 10.5)).astype(float)
    df["is_evening_ramp"] = ((hour >= 17) & (hour <= 21.5)).astype(float)
    df["abs_cong_mean"] = df["cong_mean"].abs()
    df["neg_spread_lag_48"] = -df["spread_lag_48"]
    df["neg_spread_lag_96"] = -df["spread_lag_96"]
    df["renewable_pred"] = df[["photo_gene_total_pred", "wind_gene_total_pred", "water_gene_total_pred"]].sum(axis=1)
    df["renewable_share"] = df["renewable_pred"] / df["elec_gene_total_pred"].replace(0, np.nan)
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df


def month_rank(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    vals = df[col].copy()
    if not ascending:
        vals = -vals
    return vals.groupby(df["test_month"]).rank(method="average", pct=True)


def add_conditioned_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    stress_specs = {
        "cong_abs_max": ("cong_abs_max", True),
        "cong_range": ("cong_range", True),
        "node_price_range": ("node_price_range", True),
        "abs_cong_mean": ("abs_cong_mean", True),
        "net_load_pred": ("net_load_pred", True),
        "renewable_share": ("renewable_share", True),
        "fix_out_sum": ("fix_out_sum", True),
        "machine_state_sum": ("machine_state_sum", True),
        "neg_spread_lag_48": ("neg_spread_lag_48", True),
        "neg_spread_lag_96": ("neg_spread_lag_96", True),
        "hist_tail65_rate_7d": ("hist_tail65_rate_7d", True),
        "hist_tail100_rate_7d": ("hist_tail100_rate_7d", True),
        "spread_roll_7d_min": ("spread_roll_7d_min", False),
    }
    alphas = [-0.50, -0.35, -0.20, 0.20, 0.35, 0.50, 0.70, 0.90]
    new_cols = []
    for base in BASE_SCORES:
        base_rank = month_rank(df, base, True).fillna(0.5)
        for stress_name, (col, high_risk_high_value) in stress_specs.items():
            stress_rank = month_rank(df, col, high_risk_high_value).fillna(0.5)
            for alpha in alphas:
                a_name = str(alpha).replace("-", "m").replace(".", "p")
                name = f"c9_{base}__{stress_name}_a{a_name}"
                df[name] = base_rank + alpha * stress_rank
                new_cols.append(name)
    return df, new_cols


def diagnostic_strata(df: pd.DataFrame, method: str, budget: float = 0.30) -> pd.DataFrame:
    rows = []
    p = df[df["test_month"].eq("2025-10")].copy().reset_index(drop=True)
    alert = monthly_alert_mask(p, method, budget)
    p["alert"] = alert
    features = [
        "hour",
        "is_midday",
        "is_morning_ramp",
        "is_evening_ramp",
        "cong_abs_max",
        "cong_range",
        "node_price_range",
        "net_load_pred",
        "renewable_share",
        "neg_spread_lag_48",
        "hist_tail65_rate_7d",
        "hist_tail100_rate_7d",
    ]
    for feat in features:
        s = p[feat]
        if s.nunique(dropna=True) <= 6:
            bins = s.astype(str).fillna("NA")
        else:
            bins = pd.qcut(s.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        for label, g in p.groupby(bins, dropna=False):
            mask = g["alert"].to_numpy(dtype=bool)
            met = metric_from_mask(g, mask)
            rows.append({"method": method, "budget": budget, "feature": feat, "bin": str(label), **met})
    return pd.DataFrame(rows)


def missed_event_summary(df: pd.DataFrame, methods: list[str], budget: float = 0.30) -> pd.DataFrame:
    rows = []
    for month, mdf0 in df[df["test_month"].isin(PRESSURE_MONTHS)].groupby("test_month"):
        mdf = mdf0.reset_index(drop=True)
        for method in methods:
            alert = monthly_alert_mask(mdf, method, budget)
            missed = mdf[(~alert) & (mdf["negative_tail"].astype(int).eq(1))].copy()
            hit = mdf[alert & mdf["negative_tail"].astype(int).eq(1)].copy()
            rows.append(
                {
                    "test_month": month,
                    "method": method,
                    "events": int(mdf["negative_tail"].sum()),
                    "hit_events": int(len(hit)),
                    "missed_events": int(len(missed)),
                    "missed_excess": float(missed["negative_excess"].sum()),
                    "missed_cong_abs_max_median": float(missed["cong_abs_max"].median()),
                    "hit_cong_abs_max_median": float(hit["cong_abs_max"].median()),
                    "missed_node_price_range_median": float(missed["node_price_range"].median()),
                    "hit_node_price_range_median": float(hit["node_price_range"].median()),
                    "missed_neg_spread_lag48_median": float(missed["neg_spread_lag_48"].median()),
                    "hit_neg_spread_lag48_median": float(hit["neg_spread_lag_48"].median()),
                    "missed_top_hours": str(missed["hour"].value_counts().head(8).to_dict()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_panel()
    df, conditioned = add_conditioned_scores(df)
    score_cols = BASE_SCORES + conditioned
    metrics = summarize(df, score_cols)
    metrics.to_csv(OUT / "c9_conditioned_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")

    pressure30 = metrics[(metrics["scope"].eq("pressure_available_months")) & (metrics["budget"].eq(0.30))].copy()
    pressure30 = pressure30.sort_values("gate_score", ascending=False)
    pressure30.to_csv(OUT / "c9_pressure_budget30_ranked.csv", index=False, encoding="utf-8-sig")

    key_methods = list(dict.fromkeys(BASE_SCORES + pressure30["method"].head(5).tolist()))
    missed = missed_event_summary(df, key_methods)
    missed.to_csv(OUT / "c9_pressure_missed_event_summary.csv", index=False, encoding="utf-8-sig")
    strata = pd.concat([diagnostic_strata(df, m) for m in key_methods[:8]], ignore_index=True)
    strata.to_csv(OUT / "c9_2025_10_stratified_diagnostics.csv", index=False, encoding="utf-8-sig")

    key_metrics = metrics[
        metrics["method"].isin(key_methods)
        & metrics["budget"].eq(0.30)
        & metrics["scope"].isin(["all_available_months", "pressure_available_months", "non_pressure_available_months"])
    ].sort_values(["scope", "gate_score"], ascending=[True, False])
    key_metrics.to_csv(OUT / "c9_key_metrics.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = [
        "# C9 Pressure-Month Miss Diagnosis and Conditioned Adjustment",
        "",
        "Diagnostic experiment for the 2025-10 maintenance/congestion-stress month. Conditioned scores use observable Stage-B pressure variables only, but alpha selection is diagnostic and not yet validation-safe.",
        "",
        "## Pressure 30% Best Rows",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 30),
        "",
        "## Key Metrics",
        "",
        md_table(key_metrics, ["scope", "method", "recall", "excess_share", "false_alert_burden", "gate_score"], 40),
        "",
        "## Missed Event Summary",
        "",
        md_table(missed, ["test_month", "method", "events", "hit_events", "missed_events", "missed_excess", "missed_top_hours"], 30),
    ]
    (OUT / "c9_pressure_miss_diagnosis_and_conditioned_adjustment_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
