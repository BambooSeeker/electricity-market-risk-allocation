from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c12_condition_interaction_diagnostic import monthly_rank
from c22_validation_safe_boundary_enhancement import BONUS_FEATURE, build_base_score


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "c20_distributional_tail_risk_scores" / "c20_distributional_predictions.csv"
OUT = ROOT / "c43_budget_aware_boundary_correction"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]
ALPHA = 0.20


def rank_desc_pct_by_month(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby("test_month", group_keys=False)[score_col].rank(method="first", ascending=False, pct=True)


def prepare_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_score"] = build_base_score(df)
    df["boundary_bonus"] = monthly_rank(df, BONUS_FEATURE).fillna(0.5)
    df["c22_score"] = df["base_score"] + ALPHA * df["boundary_bonus"]
    df["base_rank_pct"] = rank_desc_pct_by_month(df, "base_score")
    return df


def monthly_alert_mask_indexed(df: pd.DataFrame, score_col: str, budget: float) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, mdf in df.groupby("test_month", sort=True):
        k = max(1, int(np.ceil(len(mdf) * budget)))
        selected = mdf.sort_values(score_col, ascending=False).head(k).index
        alert.loc[selected] = True
    return alert


def fixed_protected_alert(df: pd.DataFrame, budget: float, core: float = 0.225, candidate: float = 0.40) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, mdf in df.groupby("test_month", sort=True):
        n_total = max(1, int(np.ceil(len(mdf) * budget)))
        n_core = min(int(np.floor(len(mdf) * core)), n_total)
        base_order = mdf.sort_values("base_score", ascending=False)
        core_idx = list(base_order.head(n_core).index)
        remain = n_total - len(core_idx)
        if remain <= 0:
            alert.loc[core_idx] = True
            continue
        cand = mdf.drop(index=core_idx).copy()
        cand = cand[cand["base_rank_pct"].le(candidate)]
        if len(cand) < remain:
            cand = mdf.drop(index=core_idx).copy()
        fill_idx = list(cand.sort_values("c22_score", ascending=False).head(remain).index)
        alert.loc[core_idx + fill_idx] = True
    return alert


def budget_aware_alert(df: pd.DataFrame, budget: float, band: float, candidate_extra: float) -> pd.Series:
    """Correct only a narrow base-rank boundary band.

    The protected core expands with the budget: base top (budget - band) is kept fixed,
    and mechanism-aware correction only fills the remaining band from candidates up to
    budget + candidate_extra. This targets C42's failure mode directly: broad budgets
    should not displace too many base high-risk samples.
    """
    alert = pd.Series(False, index=df.index)
    core_pct = max(0.0, budget - band)
    candidate_upper = min(1.0, budget + candidate_extra)
    for _, mdf in df.groupby("test_month", sort=True):
        n_total = max(1, int(np.ceil(len(mdf) * budget)))
        n_core = min(int(np.floor(len(mdf) * core_pct)), n_total)
        base_order = mdf.sort_values("base_score", ascending=False)
        core_idx = list(base_order.head(n_core).index)
        remain = n_total - len(core_idx)
        if remain <= 0:
            alert.loc[core_idx] = True
            continue
        cand = mdf.drop(index=core_idx).copy()
        cand = cand[cand["base_rank_pct"].le(candidate_upper)]
        if len(cand) < remain:
            cand = mdf.drop(index=core_idx).copy()
        fill_idx = list(cand.sort_values("c22_score", ascending=False).head(remain).index)
        alert.loc[core_idx + fill_idx] = True
    return alert


def eval_alert(df: pd.DataFrame, alert: pd.Series) -> dict:
    tmp = df[["test_month", "negative_tail", "negative_excess"]].copy().reset_index(drop=True)
    met = metric_from_mask(tmp, alert.loc[df.index].reset_index(drop=True))
    met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
    return met


def build_alerts(df: pd.DataFrame, budget: float) -> dict[str, pd.Series]:
    base_alert = monthly_alert_mask_indexed(df, "base_score", budget)
    fixed_alert = fixed_protected_alert(df, budget)
    band_alert = budget_aware_alert(df, budget, band=0.075, candidate_extra=0.05)
    alerts = {
        "C20b_base": base_alert,
        "C22_full_rank": monthly_alert_mask_indexed(df, "c22_score", budget),
        "protected_core_fixed": fixed_alert,
        "limited_budget_fixed_until30": fixed_alert if budget <= 0.30 else base_alert,
        "limited_budget_band_until30": band_alert if budget <= 0.30 else base_alert,
    }
    for band in [0.05, 0.075, 0.10]:
        for extra in [0.05, 0.075, 0.10]:
            name = f"budget_aware_band{str(band).replace('.', 'p')}_extra{str(extra).replace('.', 'p')}"
            alerts[name] = budget_aware_alert(df, budget, band=band, candidate_extra=extra)
    return alerts


def eval_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    for budget in BUDGETS:
        alerts = build_alerts(df, budget)
        for method, alert in alerts.items():
            for scope, sdf in scopes.items():
                rows.append({"budget": budget, "method": method, "scope": scope, **eval_alert(sdf, alert)})
    return pd.DataFrame(rows)


def delta_vs_base(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["budget", "scope"]
    base = metrics[metrics["method"].eq("C20b_base")][
        keys + ["recall", "excess_share", "false_alert_burden", "gate_score"]
    ].rename(
        columns={
            "recall": "base_recall",
            "excess_share": "base_excess_share",
            "false_alert_burden": "base_false_alert_burden",
            "gate_score": "base_gate_score",
        }
    )
    out = metrics.merge(base, on=keys, how="left")
    out["delta_recall"] = out["recall"] - out["base_recall"]
    out["delta_excess_share"] = out["excess_share"] - out["base_excess_share"]
    out["delta_gate_score"] = out["gate_score"] - out["base_gate_score"]
    return out


def method_scorecard(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pressure = delta[delta["scope"].eq("pressure_available_months")]
    for method, g in pressure.groupby("method"):
        if method == "C20b_base":
            continue
        tight = g[g["budget"].isin([0.25, 0.30])]
        broad = g[g["budget"].isin([0.35, 0.40])]
        all_budgets = g[g["budget"].isin([0.25, 0.30, 0.35, 0.40])]
        rows.append(
            {
                "method": method,
                "tight_min_delta_recall": tight["delta_recall"].min(),
                "tight_min_delta_excess_share": tight["delta_excess_share"].min(),
                "broad_min_delta_recall": broad["delta_recall"].min(),
                "broad_min_delta_excess_share": broad["delta_excess_share"].min(),
                "all_min_delta_gate_score": all_budgets["delta_gate_score"].min(),
                "tight_pass": bool((tight["delta_recall"].min() >= 0) and (tight["delta_excess_share"].min() >= 0)),
                "broad_pass": bool((broad["delta_recall"].min() >= -0.002) and (broad["delta_excess_share"].min() >= -0.002)),
                "overall_pass": bool(
                    (tight["delta_recall"].min() >= 0)
                    and (tight["delta_excess_share"].min() >= 0)
                    and (broad["delta_recall"].min() >= -0.002)
                    and (broad["delta_excess_share"].min() >= -0.002)
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["overall_pass", "tight_pass", "broad_pass", "all_min_delta_gate_score"],
        ascending=[False, False, False, False],
    )


def changed_groups(df: pd.DataFrame, method: str, budget: float) -> pd.DataFrame:
    alerts = build_alerts(df, budget)
    base = alerts["C20b_base"]
    target = alerts[method]
    pressure = df[df["test_month"].isin(PRESSURE_MONTHS)]
    rows = []
    for month, mdf in pressure.groupby("test_month", sort=True):
        groups = {
            "added": target.loc[mdf.index] & ~base.loc[mdf.index],
            "dropped": base.loc[mdf.index] & ~target.loc[mdf.index],
        }
        for group, mask in groups.items():
            g = mdf[mask]
            rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "test_month": month,
                    "group": group,
                    "n": len(g),
                    "tail_events": int(g["negative_tail"].sum()) if len(g) else 0,
                    "tail_event_rate": float(g["negative_tail"].mean()) if len(g) else np.nan,
                    "negative_excess_sum": float(g["negative_excess"].sum()) if len(g) else 0.0,
                    "base_rank_pct_median": float(g["base_rank_pct"].median()) if len(g) else np.nan,
                    "boundary_bonus_median": float(g["boundary_bonus"].median()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df = prepare_scores(df)
    metrics = eval_grid(df)
    delta = delta_vs_base(metrics)
    scorecard = method_scorecard(delta)
    best_method = scorecard.iloc[0]["method"]
    best_changes = pd.concat(
        [changed_groups(df, best_method, budget) for budget in [0.25, 0.30, 0.35, 0.40]],
        ignore_index=True,
    )

    metrics.to_csv(OUT / "c43_budget_aware_metrics.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "c43_budget_aware_delta_vs_base.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(OUT / "c43_budget_aware_scorecard.csv", index=False, encoding="utf-8-sig")
    best_changes.to_csv(OUT / "c43_best_method_added_dropped.csv", index=False, encoding="utf-8-sig")

    pressure_delta = delta[delta["scope"].eq("pressure_available_months")]
    best_pressure_delta = pressure_delta[pressure_delta["method"].eq(best_method)].sort_values("budget")
    fixed_pressure_delta = pressure_delta[pressure_delta["method"].eq("protected_core_fixed")].sort_values("budget")

    report = [
        "# C43 Budget-Aware Boundary Correction",
        "",
        "Purpose: test a principle-driven repair for C42's failure mode. The correction band is tied to alert budget: base high-risk core is protected as budget expands, and mechanism correction is restricted to a narrow boundary band.",
        "",
        f"Best diagnostic candidate by scorecard: `{best_method}`.",
        "",
        "## Method Scorecard",
        "",
        md_table(
            scorecard,
            [
                "method",
                "tight_min_delta_recall",
                "tight_min_delta_excess_share",
                "broad_min_delta_recall",
                "broad_min_delta_excess_share",
                "all_min_delta_gate_score",
                "overall_pass",
            ],
            30,
        ),
        "",
        "## Fixed Protected-Core Delta in Pressure Months",
        "",
        md_table(
            fixed_pressure_delta,
            ["budget", "delta_recall", "delta_excess_share", "delta_gate_score", "recall", "excess_share"],
            10,
        ),
        "",
        "## Best Budget-Aware Candidate Delta in Pressure Months",
        "",
        md_table(
            best_pressure_delta,
            ["budget", "delta_recall", "delta_excess_share", "delta_gate_score", "recall", "excess_share"],
            10,
        ),
        "",
        "## Best Candidate Added/Dropped Groups",
        "",
        md_table(
            best_changes,
            [
                "budget",
                "test_month",
                "group",
                "n",
                "tail_events",
                "tail_event_rate",
                "negative_excess_sum",
                "base_rank_pct_median",
                "boundary_bonus_median",
            ],
            40,
        ),
        "",
        "## Interpretation",
        "",
        "- This is a diagnostic repair, not yet a validation-safe final method.",
        "- A candidate is valuable only if it keeps the tight-budget advantage while removing the broad-budget failure.",
        "- If the best candidate works, the next step must freeze the rule using validation logic before any paper claim.",
    ]
    (OUT / "c43_budget_aware_boundary_correction_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
