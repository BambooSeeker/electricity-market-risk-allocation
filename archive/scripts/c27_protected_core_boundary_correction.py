from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c12_condition_interaction_diagnostic import monthly_rank
from c16_da_congestion_shape_diagnostic import monthly_alert_mask
from c22_validation_safe_boundary_enhancement import BONUS_FEATURE, build_base_score


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "c20_distributional_tail_risk_scores" / "c20_distributional_predictions.csv"
OUT = ROOT / "c27_protected_core_boundary_correction"
OUT.mkdir(exist_ok=True)

FINAL_BUDGET = 0.30
CORE_PCTS = [0.18, 0.20, 0.225, 0.25, 0.275]
CANDIDATE_UPPERS = [0.35, 0.40, 0.45, 0.50, 0.60]
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]


def rank01(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True).fillna(0.5)


def rank_desc_pct_by_month(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby("test_month", group_keys=False)[score_col].rank(method="first", ascending=False, pct=True)


def protected_alert_mask(
    df: pd.DataFrame,
    base_col: str,
    adjusted_col: str,
    final_budget: float,
    core_pct: float,
    candidate_upper: float,
) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, mdf in df.groupby("test_month", sort=True):
        n_total = int(np.ceil(len(mdf) * final_budget))
        n_core = min(int(np.floor(len(mdf) * core_pct)), n_total)
        base_order = mdf.sort_values(base_col, ascending=False)
        core_idx = list(base_order.head(n_core).index)
        remaining_slots = n_total - len(core_idx)
        if remaining_slots <= 0:
            alert.loc[core_idx] = True
            continue
        candidate = mdf.drop(index=core_idx).copy()
        candidate = candidate[candidate["base_rank_pct"].le(candidate_upper)]
        if len(candidate) < remaining_slots:
            candidate = mdf.drop(index=core_idx).copy()
        fill_idx = list(candidate.sort_values(adjusted_col, ascending=False).head(remaining_slots).index)
        alert.loc[core_idx + fill_idx] = True
    return alert


def eval_one(df: pd.DataFrame, alert: pd.Series) -> dict:
    tmp = df[["test_month", "negative_tail", "negative_excess"]].copy().reset_index(drop=True)
    if not isinstance(alert, pd.Series):
        alert = pd.Series(alert, index=df.index)
    met = metric_from_mask(tmp, alert.loc[df.index].reset_index(drop=True))
    met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
    return met


def eval_scopes(df: pd.DataFrame, alert: pd.Series) -> list[dict]:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    for scope, sdf in scopes.items():
        rows.append({"scope": scope, **eval_one(sdf, alert)})
    return rows


def monthly_breakdown(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for month, mdf in df.groupby("test_month", sort=True):
        for name, alert in alerts.items():
            rows.append({"test_month": month, "method": name, **eval_one(mdf, alert)})
    return pd.DataFrame(rows)


def changed_summary(df: pd.DataFrame, base_alert: pd.Series, method_alert: pd.Series, method: str) -> pd.DataFrame:
    rows = []
    for month, mdf in df[df["test_month"].isin(PRESSURE_MONTHS)].groupby("test_month", sort=True):
        local_base = base_alert.loc[mdf.index]
        local_method = method_alert.loc[mdf.index]
        groups = {
            "added": local_method & ~local_base,
            "dropped": local_base & ~local_method,
        }
        for group, mask in groups.items():
            g = mdf[mask]
            rows.append(
                {
                    "test_month": month,
                    "method": method,
                    "group": group,
                    "n": len(g),
                    "tail_events": int(g["negative_tail"].sum()) if len(g) else 0,
                    "negative_excess_sum": float(g["negative_excess"].sum()) if len(g) else 0.0,
                    "bonus_median": float(g[BONUS_FEATURE].median()) if len(g) else np.nan,
                    "base_rank_pct_median": float(g["base_rank_pct"].median()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df["base_score"] = build_base_score(df)
    df["boundary_bonus"] = monthly_rank(df, BONUS_FEATURE).fillna(0.5)
    df["base_rank_pct"] = rank_desc_pct_by_month(df, "base_score")

    base_alert = monthly_alert_mask(df, "base_score", FINAL_BUDGET)
    base_alert = pd.Series(base_alert, index=df.index)
    full_alerts: dict[str, pd.Series] = {"C20b_base": base_alert}
    metric_rows = []
    for row in eval_scopes(df, base_alert):
        metric_rows.append({"method": "C20b_base", "core_pct": np.nan, "candidate_upper": np.nan, "alpha": 0.0, **row})

    for alpha in ALPHAS:
        adj_col = f"adjusted_alpha_{str(alpha).replace('.', 'p')}"
        df[adj_col] = df["base_score"] + alpha * df["boundary_bonus"]
        full_rank_alert = monthly_alert_mask(df, adj_col, FINAL_BUDGET)
        full_rank_alert = pd.Series(full_rank_alert, index=df.index)
        name = f"full_rank_alpha_{str(alpha).replace('.', 'p')}"
        full_alerts[name] = full_rank_alert
        for row in eval_scopes(df, full_rank_alert):
            metric_rows.append({"method": name, "core_pct": 0.0, "candidate_upper": 1.0, "alpha": alpha, **row})

        for core_pct in CORE_PCTS:
            for candidate_upper in CANDIDATE_UPPERS:
                if candidate_upper <= core_pct or core_pct >= FINAL_BUDGET:
                    continue
                alert = protected_alert_mask(df, "base_score", adj_col, FINAL_BUDGET, core_pct, candidate_upper)
                name = (
                    f"protected_core_{str(core_pct).replace('.', 'p')}"
                    f"_cand_{str(candidate_upper).replace('.', 'p')}"
                    f"_alpha_{str(alpha).replace('.', 'p')}"
                )
                full_alerts[name] = alert
                for row in eval_scopes(df, alert):
                    metric_rows.append(
                        {
                            "method": name,
                            "core_pct": core_pct,
                            "candidate_upper": candidate_upper,
                            "alpha": alpha,
                            **row,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics["pressure_hard_gate"] = (
        metrics["scope"].eq("pressure_available_months")
        & metrics["recall"].ge(0.70)
        & metrics["excess_share"].ge(0.75)
    )
    metrics["all_hard_gate"] = (
        metrics["scope"].eq("all_available_months")
        & metrics["recall"].ge(0.75)
        & metrics["excess_share"].ge(0.80)
    )
    metrics.to_csv(OUT / "c27_metrics.csv", index=False, encoding="utf-8-sig")

    pressure = metrics[metrics["scope"].eq("pressure_available_months")].copy()
    pressure = pressure.sort_values(["pressure_hard_gate", "gate_score", "recall", "excess_share"], ascending=[False, False, False, False])
    all_month = metrics[metrics["scope"].eq("all_available_months")].copy()
    all_month = all_month.sort_values(["all_hard_gate", "gate_score", "recall", "excess_share"], ascending=[False, False, False, False])

    selected_methods = list(dict.fromkeys(["C20b_base", "full_rank_alpha_0p2"] + pressure["method"].head(5).tolist()))
    selected_alerts = {k: full_alerts[k] for k in selected_methods if k in full_alerts}
    monthly = monthly_breakdown(df, selected_alerts)
    changed = pd.concat(
        [changed_summary(df, base_alert, selected_alerts[m], m) for m in selected_methods if m != "C20b_base"],
        ignore_index=True,
    )
    pressure.to_csv(OUT / "c27_pressure_ranked.csv", index=False, encoding="utf-8-sig")
    all_month.to_csv(OUT / "c27_all_ranked.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "c27_monthly_breakdown.csv", index=False, encoding="utf-8-sig")
    changed.to_csv(OUT / "c27_pressure_changed_summary.csv", index=False, encoding="utf-8-sig")

    report = [
        "# C27 Protected-Core Boundary Correction",
        "",
        "Purpose: reduce the event-level weakness found in C26, where full-rank mechanism correction can displace some high-base-risk tail events.",
        "",
        "## Pressure-Month Ranking",
        "",
        md_table(
            pressure,
            ["method", "core_pct", "candidate_upper", "alpha", "recall", "excess_share", "false_alert_burden", "gate_score", "pressure_hard_gate"],
            40,
        ),
        "",
        "## All-Month Ranking",
        "",
        md_table(
            all_month,
            ["method", "core_pct", "candidate_upper", "alpha", "recall", "excess_share", "false_alert_burden", "gate_score", "all_hard_gate"],
            40,
        ),
        "",
        "## Monthly Breakdown for Selected Methods",
        "",
        md_table(
            monthly,
            ["test_month", "method", "recall", "excess_share", "false_alert_burden", "gate_score"],
            60,
        ),
        "",
        "## Pressure Changed Groups vs Base",
        "",
        md_table(
            changed,
            ["test_month", "method", "group", "n", "tail_events", "negative_excess_sum", "bonus_median", "base_rank_pct_median"],
            80,
        ),
    ]
    (OUT / "c27_protected_core_boundary_correction_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
