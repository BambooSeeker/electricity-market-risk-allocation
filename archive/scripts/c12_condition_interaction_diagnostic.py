from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import BASE_SCORES, build_panel, metric_from_mask


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c12_condition_interaction_diagnostic"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
BASES = [
    "AllSignals_equal_rank_avg",
    "HGB_q10_distance_two_stage",
    "QIA_two_stage",
    "tcn_s48_h64_w6_vmd_hgb_blend",
    "tcn_s48_h64_w6_vmd_qia_blend",
]


def monthly_rank(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    vals = df[col].copy()
    if not ascending:
        vals = -vals
    return vals.groupby(df["test_month"]).rank(method="average", pct=True).fillna(0.5)


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_specs = {
        "machine": ("machine_state_sum", True),
        "neg_lag48": ("neg_spread_lag_48", True),
        "neg_lag96": ("neg_spread_lag_96", True),
        "hist65": ("hist_tail65_rate_7d", True),
        "hist100": ("hist_tail100_rate_7d", True),
        "cong_abs": ("cong_abs_max", True),
        "cong_range": ("cong_range", True),
        "node_range": ("node_price_range", True),
        "net_load": ("net_load_pred", True),
        "renewable_share": ("renewable_share", True),
    }
    for name, (col, high) in raw_specs.items():
        df[f"r_{name}"] = monthly_rank(df, col, ascending=high)
        r = df[f"r_{name}"]
        df[f"mid_{name}"] = (1.0 - (r - 0.5).abs() * 2.0).clip(0, 1)
        df[f"hi_{name}"] = (r >= 0.75).astype(float)
        df[f"lo_{name}"] = (r <= 0.25).astype(float)
    hour = df["time"].dt.hour
    df["r_midday"] = (((hour >= 10) & (hour <= 15)).astype(float)).groupby(df["test_month"]).rank(method="average", pct=True)
    df["r_ramp"] = (((hour >= 6) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))).astype(float)
    return df


def alert_mask_monthly(df: pd.DataFrame, score: np.ndarray, budget: float) -> np.ndarray:
    alert = np.zeros(len(df), dtype=bool)
    score_s = pd.Series(score, index=df.index)
    for _, idx in df.groupby("test_month", sort=True).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        order = np.argsort(-score_s.loc[idx_arr].to_numpy())
        k = max(1, int(np.ceil(len(idx_arr) * budget)))
        alert[idx_arr[order[:k]]] = True
    return alert


def summarize_score(df: pd.DataFrame, score: np.ndarray, name: str) -> list[dict]:
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    rows = []
    score_s = pd.Series(score, index=df.index)
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        s = score_s.loc[sdf.index].to_numpy()
        sdf_eval = sdf.reset_index(drop=True)
        for budget in BUDGETS:
            alert = alert_mask_monthly(sdf_eval, s, budget)
            met = metric_from_mask(sdf_eval, alert)
            rows.append({"scope": scope, "method": name, "budget": budget, **met})
    return rows


def main() -> None:
    df = add_state_features(build_panel())
    state_terms = [
        "r_machine",
        "r_neg_lag48",
        "r_neg_lag96",
        "r_hist65",
        "r_hist100",
        "r_cong_abs",
        "r_cong_range",
        "r_node_range",
        "mid_node_range",
        "mid_hist65",
        "mid_cong_abs",
        "r_midday",
        "r_ramp",
    ]
    pair_terms = [
        ("r_machine", "r_neg_lag96"),
        ("r_machine", "r_hist65"),
        ("r_cong_abs", "r_node_range"),
        ("r_cong_range", "r_node_range"),
        ("r_hist65", "r_cong_abs"),
        ("r_hist65", "mid_node_range"),
        ("mid_hist65", "mid_node_range"),
        ("r_ramp", "r_cong_abs"),
        ("r_midday", "r_hist65"),
    ]
    alphas = [-0.6, -0.35, -0.2, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1]
    gammas = [-0.8, -0.45, -0.25, 0.25, 0.45, 0.7, 0.95]
    rows = []
    score_cache: dict[str, np.ndarray] = {}

    for base in BASES:
        base_rank = monthly_rank(df, base).to_numpy()
        for t1 in state_terms:
            v1 = df[t1].fillna(0.5).to_numpy()
            for a in alphas:
                name = f"c12_{base}__{t1}_a{a:+.2f}"
                score = base_rank + a * v1
                for r in summarize_score(df, score, name):
                    if r["scope"] == "pressure_available_months" and r["budget"] == 0.30:
                        r["gate_score"] = r["recall"] + r["excess_share"] - 0.2 * r["false_alert_burden"]
                        rows.append(r)
                score_cache[name] = score
        for t1, t2 in pair_terms:
            v1 = df[t1].fillna(0.5).to_numpy()
            v2 = df[t2].fillna(0.5).to_numpy()
            inter = v1 * v2
            for a in [0.25, 0.45, 0.65, 0.85]:
                for b in [0.25, 0.45, 0.65, 0.85]:
                    for g in gammas:
                        name = f"c12_{base}__{t1}_{a:.2f}__{t2}_{b:.2f}__ix{g:+.2f}"
                        score = base_rank + a * v1 + b * v2 + g * inter
                        for r in summarize_score(df, score, name):
                            if r["scope"] == "pressure_available_months" and r["budget"] == 0.30:
                                r["gate_score"] = r["recall"] + r["excess_share"] - 0.2 * r["false_alert_burden"]
                                rows.append(r)
                        score_cache[name] = score

    pressure = pd.DataFrame(rows).sort_values("gate_score", ascending=False).reset_index(drop=True)
    pressure.to_csv(OUT / "c12_pressure_budget30_diagnostic_ranked.csv", index=False, encoding="utf-8-sig")
    best_names = list(dict.fromkeys(BASES + pressure["method"].head(20).tolist()))
    all_rows = []
    for base in BASES:
        all_rows.extend(summarize_score(df, monthly_rank(df, base).to_numpy(), base))
    for name in best_names:
        if name in score_cache:
            all_rows.extend(summarize_score(df, score_cache[name], name))
    metrics = pd.DataFrame(all_rows)
    metrics["gate_score"] = metrics["recall"] + metrics["excess_share"] - 0.2 * metrics["false_alert_burden"]
    metrics.to_csv(OUT / "c12_key_metrics.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    key = metrics[
        metrics["budget"].eq(0.30)
        & metrics["scope"].isin(["all_available_months", "pressure_available_months", "non_pressure_available_months"])
    ].sort_values(["scope", "gate_score"], ascending=[True, False])
    report = [
        "# C12 Condition-Interaction Diagnostic",
        "",
        "Diagnostic search for observable maintenance/congestion-stress interactions. This is not yet validation-safe model selection; it tests whether interaction structure can close the pressure-month hard-gate gap.",
        "",
        "## Pressure 30% Best Diagnostic Rows",
        "",
        md_table(pressure, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 40),
        "",
        "## Key Metrics for Best Rows",
        "",
        md_table(key, ["scope", "method", "recall", "excess_share", "false_alert_burden", "gate_score"], 60),
    ]
    (OUT / "c12_condition_interaction_diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
