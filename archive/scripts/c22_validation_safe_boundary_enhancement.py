from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c12_condition_interaction_diagnostic import monthly_rank
from c16_da_congestion_shape_diagnostic import monthly_alert_mask


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "c20_distributional_tail_risk_scores" / "c20_distributional_predictions.csv"
OUT = ROOT / "c22_validation_safe_boundary_enhancement"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
ALPHAS = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
BASE_EXPR = (
    "c20_blend_c20_hgb_tail_prob_cls__C13_best_exact_w0p2|0.6"
    "+c20_blend_c20_iso_tail_prob__C19_high_recall_pair_proxy_w0p2|0.4"
)
BONUS_FEATURE = "da_cong_negative_pressure_delta48"


def rank01(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True).fillna(0.5)


def parse_fusion(expr: str) -> list[tuple[str, float]]:
    out = []
    for part in expr.split("+"):
        name, weight = part.rsplit("|", 1)
        out.append((name, float(weight)))
    return out


def build_base_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, weight in parse_fusion(BASE_EXPR):
        score = score + weight * rank01(df[col])
    return score


def eval_score(df: pd.DataFrame, score: pd.Series | np.ndarray, method: str) -> pd.DataFrame:
    tmp = df[["test_month", "negative_tail", "negative_excess"]].copy().reset_index(drop=True)
    tmp["_score"] = np.asarray(score)
    rows = []
    scopes = {
        "all_available_months": tmp,
        "pressure_available_months": tmp[tmp["test_month"].isin(PRESSURE_MONTHS)].reset_index(drop=True),
        "non_pressure_available_months": tmp[~tmp["test_month"].isin(PRESSURE_MONTHS)].reset_index(drop=True),
    }
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for budget in BUDGETS:
            alert = monthly_alert_mask(sdf, "_score", budget)
            rows.append({"scope": scope, "method": method, "budget": budget, **metric_from_mask(sdf, alert)})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    out["pressure_hard_gate"] = (
        out["scope"].eq("pressure_available_months")
        & out["budget"].eq(0.30)
        & out["recall"].ge(0.70)
        & out["excess_share"].ge(0.75)
    )
    out["all_hard_gate"] = (
        out["scope"].eq("all_available_months")
        & out["budget"].eq(0.30)
        & out["recall"].ge(0.75)
        & out["excess_share"].ge(0.80)
    )
    return out


def per_month_metric(df: pd.DataFrame, score: pd.Series | np.ndarray, method: str, budget: float = 0.30) -> pd.DataFrame:
    tmp = df[["test_month", "negative_tail", "negative_excess"]].copy().reset_index(drop=True)
    tmp["_score"] = np.asarray(score)
    rows = []
    for month, mdf0 in tmp.groupby("test_month", sort=True):
        mdf = mdf0.reset_index(drop=True)
        alert = monthly_alert_mask(mdf, "_score", budget)
        rows.append({"test_month": month, "method": method, **metric_from_mask(mdf, alert)})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def select_alpha_on_frame(df: pd.DataFrame, base: pd.Series, bonus: pd.Series, rule: str) -> float:
    rows = []
    for alpha in ALPHAS:
        met = eval_score(df, base + alpha * bonus, f"alpha_{alpha:+.2f}")
        row = met[met["scope"].eq("all_available_months") & met["budget"].eq(0.30)].iloc[0].to_dict()
        row["alpha"] = alpha
        rows.append(row)
    cand = pd.DataFrame(rows)
    if rule == "gate_score":
        cand = cand.sort_values(["gate_score", "recall", "excess_share"], ascending=False)
    elif rule == "recall_with_excess70":
        tmp = cand[cand["excess_share"].ge(0.70)]
        if tmp.empty:
            tmp = cand
        cand = tmp.sort_values(["recall", "excess_share", "gate_score"], ascending=False)
    elif rule == "recall_with_excess75":
        tmp = cand[cand["excess_share"].ge(0.75)]
        if tmp.empty:
            tmp = cand
        cand = tmp.sort_values(["recall", "excess_share", "gate_score"], ascending=False)
    else:
        raise ValueError(rule)
    return float(cand.iloc[0]["alpha"])


def temporal_adaptive_score(df: pd.DataFrame, base: pd.Series, bonus: pd.Series, default_alpha: float, rule: str) -> tuple[pd.Series, pd.DataFrame]:
    months = sorted(df["test_month"].unique())
    score = pd.Series(np.nan, index=df.index)
    choices = []
    for i, month in enumerate(months):
        if i == 0:
            alpha = default_alpha
            source = "default"
        else:
            prev_month = months[i - 1]
            prev_idx = df["test_month"].eq(prev_month)
            alpha = select_alpha_on_frame(
                df.loc[prev_idx].reset_index(drop=True),
                base.loc[prev_idx].reset_index(drop=True),
                bonus.loc[prev_idx].reset_index(drop=True),
                rule,
            )
            source = prev_month
        idx = df["test_month"].eq(month)
        score.loc[idx] = base.loc[idx] + alpha * bonus.loc[idx]
        choices.append({"test_month": month, "alpha": alpha, "source": source, "rule": rule, "default_alpha": default_alpha})
    return score, pd.DataFrame(choices)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    base = build_base_score(df)
    bonus = monthly_rank(df, BONUS_FEATURE).fillna(0.5)

    metric_parts = []
    month_parts = []
    choice_parts = []

    fixed_specs = {
        "C22_base_no_bonus": 0.0,
        "C22_mechanism_fixed_alpha_0p1": 0.10,
        "C22_mechanism_fixed_alpha_0p2": 0.20,
        "C22_mechanism_fixed_alpha_0p3": 0.30,
    }
    for name, alpha in fixed_specs.items():
        score = base + alpha * bonus
        metric_parts.append(eval_score(df, score, name))
        month_parts.append(per_month_metric(df, score, name))

    # Chronological rules: alpha for each month is selected using the immediately previous evaluated month only.
    for default_alpha in [0.0, 0.10, 0.20]:
        for rule in ["gate_score", "recall_with_excess70", "recall_with_excess75"]:
            name = f"C22_temporal_prevmonth_{rule}_default_{str(default_alpha).replace('.', 'p')}"
            score, choices = temporal_adaptive_score(df, base, bonus, default_alpha, rule)
            choices["method"] = name
            choice_parts.append(choices)
            metric_parts.append(eval_score(df, score, name))
            month_parts.append(per_month_metric(df, score, name))

    # Non-pressure transfer is diagnostic, not chronological; it asks whether the same alpha direction is supported outside pressure months.
    nonpressure_idx = ~df["test_month"].isin(PRESSURE_MONTHS)
    for rule in ["gate_score", "recall_with_excess70", "recall_with_excess75"]:
        alpha = select_alpha_on_frame(
            df.loc[nonpressure_idx].reset_index(drop=True),
            base.loc[nonpressure_idx].reset_index(drop=True),
            bonus.loc[nonpressure_idx].reset_index(drop=True),
            rule,
        )
        name = f"C22_nonpressure_transfer_{rule}_alpha_{str(alpha).replace('-', 'm').replace('.', 'p')}"
        score = base + alpha * bonus
        metric_parts.append(eval_score(df, score, name))
        month_parts.append(per_month_metric(df, score, name))
        choice_parts.append(pd.DataFrame([{"method": name, "alpha": alpha, "source": "non_pressure_months", "rule": rule}]))

    metrics = pd.concat(metric_parts, ignore_index=True)
    monthly = pd.concat(month_parts, ignore_index=True)
    choices = pd.concat(choice_parts, ignore_index=True)
    metrics.to_csv(OUT / "c22_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "c22_monthly_budget30.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "c22_alpha_choices.csv", index=False, encoding="utf-8-sig")

    pressure30 = metrics[metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)].copy()
    pressure30 = pressure30.sort_values(["pressure_hard_gate", "gate_score"], ascending=[False, False])
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].copy()
    all30 = all30.sort_values(["all_hard_gate", "gate_score"], ascending=[False, False])

    selected_methods = list(dict.fromkeys(pressure30["method"].head(8).tolist() + all30["method"].head(5).tolist()))
    selected_monthly = monthly[monthly["method"].isin(selected_methods)].sort_values(["method", "test_month"])

    report = [
        "# C22 Validation-Safe Boundary Enhancement",
        "",
        "C22 tests whether the C21 boundary variable can be fixed or selected without directly tuning on the locked pressure-month aggregate. The mechanism-fixed alpha=0.2 result is confirmatory but should still be reported separately from strict chronological selection.",
        "",
        f"Base score: `{BASE_EXPR}`",
        "",
        f"Boundary variable: `{BONUS_FEATURE}`",
        "",
        "## Pressure 30% Results",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "pressure_hard_gate"], 25),
        "",
        "## All-Month 30% Results",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "all_hard_gate"], 20),
        "",
        "## Alpha Choices",
        "",
        md_table(choices, [c for c in ["method", "test_month", "alpha", "source", "rule", "default_alpha"] if c in choices.columns], 40),
        "",
        "## Selected Monthly Breakdown",
        "",
        md_table(selected_monthly, ["test_month", "method", "recall", "excess_share", "false_alert_burden", "gate_score"], 80),
    ]
    (OUT / "c22_validation_safe_boundary_enhancement_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
