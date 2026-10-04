from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import build_panel, metric_from_mask
from c12_condition_interaction_diagnostic import add_state_features, monthly_rank


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c16_da_congestion_shape_diagnostic"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
BASE_SCORES = [
    "AllSignals_equal_rank_avg",
    "HGB_q10_distance_two_stage",
    "QIA_two_stage",
    "tcn_s48_h64_w6_vmd_hgb_blend",
    "tcn_s48_h64_w6_vmd_qia_blend",
]


def monthly_alert_mask(df: pd.DataFrame, score_col: str, budget: float) -> np.ndarray:
    alert = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("test_month", sort=True).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        scores = df.loc[idx_arr, score_col].fillna(-np.inf).to_numpy()
        order = np.argsort(-scores)
        k = max(1, int(np.ceil(len(idx_arr) * budget)))
        alert[idx_arr[order[:k]]] = True
    return alert


def summarize(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    for scope, sdf0 in scopes.items():
        sdf = sdf0.reset_index(drop=True)
        if sdf.empty:
            continue
        for score in score_cols:
            for budget in BUDGETS:
                alert = monthly_alert_mask(sdf, score, budget)
                rows.append({"scope": scope, "method": score, "budget": budget, **metric_from_mask(sdf, alert)})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    out["hard_gate_pressure30"] = (
        out["scope"].eq("pressure_available_months")
        & out["budget"].eq(0.30)
        & out["recall"].ge(0.70)
        & out["excess_share"].ge(0.75)
    )
    return out


def add_da_congestion_shape(df: pd.DataFrame) -> pd.DataFrame:
    df = add_state_features(df).copy()
    cong_cols = [
        c
        for c in df.columns
        if c.startswith("price_day_ahead_cong_") and df[c].notna().any()
    ]
    if not cong_cols:
        raise RuntimeError("No usable price_day_ahead_cong_* columns found.")

    mat = df[cong_cols].astype(float)
    df["da_cong_node_mean"] = mat.mean(axis=1)
    df["da_cong_node_abs_mean"] = mat.abs().mean(axis=1)
    df["da_cong_node_abs_max"] = mat.abs().max(axis=1)
    df["da_cong_node_std"] = mat.std(axis=1)
    df["da_cong_node_range"] = mat.max(axis=1) - mat.min(axis=1)
    df["da_cong_node_min"] = mat.min(axis=1)
    df["da_cong_node_max"] = mat.max(axis=1)
    df["da_cong_negative_pressure"] = (-mat.clip(upper=0)).mean(axis=1)
    df["da_cong_positive_pressure"] = mat.clip(lower=0).mean(axis=1)
    df["da_cong_signed_imbalance"] = df["da_cong_positive_pressure"] - df["da_cong_negative_pressure"]

    for col in [
        "da_cong_node_mean",
        "da_cong_node_abs_mean",
        "da_cong_node_abs_max",
        "da_cong_node_std",
        "da_cong_node_range",
        "da_cong_negative_pressure",
        "da_cong_positive_pressure",
        "da_cong_signed_imbalance",
    ]:
        df[f"{col}_lag48"] = df[col].shift(48)
        df[f"{col}_delta48"] = df[col] - df[f"{col}_lag48"]
        df[f"{col}_roll7_abs"] = df[col].shift(48).abs().rolling(336, min_periods=48).mean()

    rank_specs = {
        "r_da_abs": ("da_cong_node_abs_mean", True),
        "r_da_absmax": ("da_cong_node_abs_max", True),
        "r_da_std": ("da_cong_node_std", True),
        "r_da_range": ("da_cong_node_range", True),
        "r_da_neg": ("da_cong_negative_pressure", True),
        "r_da_pos": ("da_cong_positive_pressure", True),
        "r_da_imb_abs": ("da_cong_signed_imbalance", True),
        "r_da_delta_abs": ("da_cong_node_abs_mean_delta48", True),
        "r_da_delta_neg": ("da_cong_negative_pressure_delta48", True),
        "r_da_roll_abs": ("da_cong_node_abs_mean_roll7_abs", True),
        "r_da_roll_neg": ("da_cong_negative_pressure_roll7_abs", True),
    }
    for name, (col, high_risk_high_value) in rank_specs.items():
        df[name] = monthly_rank(df, col, high_risk_high_value).fillna(0.5)

    df["C12_severity_best"] = (
        monthly_rank(df, "HGB_q10_distance_two_stage").fillna(0.5)
        + 0.65 * df["r_machine"].fillna(0.5)
        + 0.25 * df["r_neg_lag96"].fillna(0.5)
        + 0.25 * df["r_machine"].fillna(0.5) * df["r_neg_lag96"].fillna(0.5)
    )
    df["C13_best_exact"] = (
        0.35 * monthly_rank(df, "tcn_s48_h64_w6_vmd_hgb_blend").fillna(0.5)
        + 0.65 * monthly_rank(df, "C12_severity_best").fillna(0.5)
    )

    df["da_cong_memory_interaction"] = df["r_da_abs"] * df["r_neg_lag96"].fillna(0.5)
    df["da_cong_machine_interaction"] = df["r_da_abs"] * df["r_machine"].fillna(0.5)
    df["da_neg_memory_interaction"] = df["r_da_neg"] * df["r_neg_lag48"].fillna(0.5)
    df["da_delta_state_interaction"] = df["r_da_delta_abs"] * df["r_machine"].fillna(0.5)
    return df


def add_candidate_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    signals = [
        "r_da_abs",
        "r_da_absmax",
        "r_da_std",
        "r_da_range",
        "r_da_neg",
        "r_da_pos",
        "r_da_imb_abs",
        "r_da_delta_abs",
        "r_da_delta_neg",
        "r_da_roll_abs",
        "r_da_roll_neg",
        "da_cong_memory_interaction",
        "da_cong_machine_interaction",
        "da_neg_memory_interaction",
        "da_delta_state_interaction",
    ]
    bases = BASE_SCORES + ["C12_severity_best", "C13_best_exact"]
    alphas = [-0.30, -0.15, 0.15, 0.25, 0.35, 0.50, 0.70]
    new_cols = []
    for base in bases:
        if base not in df:
            continue
        b = monthly_rank(df, base).fillna(0.5)
        for sig in signals:
            s = df[sig].fillna(0.5)
            for alpha in alphas:
                tag = str(alpha).replace("-", "m").replace(".", "p")
                name = f"c16_{base}__{sig}_a{tag}"
                df[name] = b + alpha * s
                new_cols.append(name)

    # A small set of two-signal variants, kept interpretable and mechanism-bound.
    combos = {
        "da_abs_memory": 0.45 * df["r_da_abs"].fillna(0.5) + 0.35 * df["r_neg_lag96"].fillna(0.5),
        "da_neg_lag48_machine": 0.40 * df["r_da_neg"].fillna(0.5) + 0.35 * df["r_neg_lag48"].fillna(0.5) + 0.30 * df["r_machine"].fillna(0.5),
        "da_delta_machine": 0.45 * df["r_da_delta_abs"].fillna(0.5) + 0.40 * df["r_machine"].fillna(0.5),
    }
    for combo_name, combo_score in combos.items():
        df[f"c16_signal_{combo_name}"] = combo_score
        new_cols.append(f"c16_signal_{combo_name}")
        for base in ["C12_severity_best", "C13_best_exact", "AllSignals_equal_rank_avg"]:
            b = monthly_rank(df, base).fillna(0.5)
            for alpha in [0.20, 0.35, 0.50]:
                tag = str(alpha).replace(".", "p")
                name = f"c16_{base}__{combo_name}_a{tag}"
                df[name] = b + alpha * monthly_rank(df.assign(_combo=combo_score), "_combo").fillna(0.5)
                new_cols.append(name)
    return df, new_cols


def per_pressure_month(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for month, mdf0 in df[df["test_month"].isin(PRESSURE_MONTHS)].groupby("test_month"):
        mdf = mdf0.reset_index(drop=True)
        for method in methods:
            alert = monthly_alert_mask(mdf, method, 0.30)
            rows.append({"test_month": month, "method": method, **metric_from_mask(mdf, alert)})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = add_da_congestion_shape(build_panel()).sort_values("time").reset_index(drop=True)
    df, candidates = add_candidate_scores(df)
    score_cols = list(dict.fromkeys(BASE_SCORES + ["C12_severity_best", "C13_best_exact"] + candidates))
    metrics = summarize(df, score_cols)
    metrics.to_csv(OUT / "c16_metrics.csv", index=False, encoding="utf-8-sig")
    pressure30 = metrics[
        metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)
    ].sort_values(["hard_gate_pressure30", "gate_score"], ascending=[False, False])
    pressure30.to_csv(OUT / "c16_pressure_budget30_ranked.csv", index=False, encoding="utf-8-sig")
    key_methods = list(dict.fromkeys(["AllSignals_equal_rank_avg", "C12_severity_best", "C13_best_exact"] + pressure30["method"].head(12).tolist()))
    month_breakdown = per_pressure_month(df, key_methods)
    month_breakdown.to_csv(OUT / "c16_pressure_month_breakdown.csv", index=False, encoding="utf-8-sig")

    report = [
        "# C16 Day-Ahead Congestion Shape Diagnostic",
        "",
        "This diagnostic tests whether node-level day-ahead congestion components provide mechanism-consistent incremental information for pressure-month negative-tail risk ranking.",
        "",
        "## Pressure 30% Best Rows",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "hard_gate_pressure30"], 40),
        "",
        "## Pressure-Month Breakdown for Top Methods",
        "",
        md_table(month_breakdown.sort_values(["test_month", "method"]), ["test_month", "method", "recall", "excess_share", "false_alert_burden"], 80),
    ]
    (OUT / "c16_da_congestion_shape_diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
