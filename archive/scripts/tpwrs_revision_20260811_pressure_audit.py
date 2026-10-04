from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import build_alerts as build_base_alerts
from c43_budget_aware_boundary_correction import eval_alert, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED
from c49_pressure_transfer_safe_candidate_selection import candidate_alert
from c77_stress_gated_deep_boundary_enhancement import (
    BUDGET,
    C50,
    stress_gated_alert,
    threshold_from_validation,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tpwrs_revision_20260811"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
RNG = np.random.default_rng(20260811)


def build_alerts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    scored = prepare_scores(df.copy())
    core_preserved = build_base_alerts(scored, BUDGET)["limited_budget_band_until30"]
    unrestricted = candidate_alert(scored, C50, BUDGET)
    threshold = threshold_from_validation(scored, "net_load_pred", 0.60)
    gated = stress_gated_alert(scored, "net_load_pred", threshold, 0.05, 0.50)
    return core_preserved, unrestricted, gated


def primary_metrics(df: pd.DataFrame, alert: pd.Series) -> dict[str, float]:
    result = eval_alert(df, alert.loc[df.index])
    return {
        "recall": float(result["recall"]),
        "excess_share": float(result["excess_share"]),
    }


def month_audit(
    df: pd.DataFrame, comparators: dict[str, pd.Series], gated: pd.Series
) -> pd.DataFrame:
    rows = []
    for comparator, reference in comparators.items():
        for month, month_df in df.groupby("test_month", sort=True):
            base = primary_metrics(month_df, reference)
            gate = primary_metrics(month_df, gated)
            changed = int((reference.loc[month_df.index] != gated.loc[month_df.index]).sum())
            rows.append(
                {
                    "comparator": comparator,
                    "month": month,
                    "intervals": len(month_df),
                    "events": int(month_df["negative_tail"].sum()),
                    "membership_changes": changed,
                    "exchanges": changed // 2,
                    "reference_recall": base["recall"],
                    "gated_recall": gate["recall"],
                    "delta_recall": gate["recall"] - base["recall"],
                    "reference_excess_share": base["excess_share"],
                    "gated_excess_share": gate["excess_share"],
                    "delta_excess_share": gate["excess_share"] - base["excess_share"],
                }
            )
    return pd.DataFrame(rows)


def paired_day_bootstrap(
    df: pd.DataFrame, comparators: dict[str, pd.Series], gated: pd.Series
) -> pd.DataFrame:
    pressure = df[df["test_month"].isin(PRESSURE_MONTHS)].copy()
    pressure["day"] = pressure["time"].dt.date.astype(str)
    days = [part.index.to_numpy() for _, part in pressure.groupby("day", sort=True)]
    rows = []
    for draw in range(N_BOOT):
        selected = RNG.integers(0, len(days), size=len(days))
        indices = np.concatenate([days[i] for i in selected])
        sample = df.loc[indices].reset_index(drop=True)
        gate_alert = gated.loc[indices].reset_index(drop=True)
        gate = primary_metrics(sample, gate_alert)
        for comparator, reference in comparators.items():
            base_alert = reference.loc[indices].reset_index(drop=True)
            base = primary_metrics(sample, base_alert)
            rows.append(
                {
                    "draw": draw,
                    "comparator": comparator,
                    "delta_recall": gate["recall"] - base["recall"],
                    "delta_excess_share": gate["excess_share"] - base["excess_share"],
                }
            )
    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator, group in samples.groupby("comparator", sort=True):
        for metric in ["delta_recall", "delta_excess_share"]:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "mean": values.mean(),
                    "p025": np.percentile(values, 2.5),
                    "p975": np.percentile(values, 97.5),
                    "probability_positive": (values > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    core_preserved, unrestricted, gated = build_alerts(df)
    comparators = {
        "core_preserved_starting_set": core_preserved,
        "unrestricted_qmlp": unrestricted,
    }
    pressure = df[df["test_month"].isin(PRESSURE_MONTHS)]

    months = month_audit(pressure, comparators, gated)
    samples = paired_day_bootstrap(df, comparators, gated)
    summary = summarize(samples)

    months.to_csv(OUT / "pressure_month_dual_metric_audit.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT / "pressure_day_block_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "pressure_day_block_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    report = [
        "# TPWRS pressure-gate evidence audit",
        "",
        "This audit uses frozen confirmation predictions and does not retrain any model.",
        "Recall and ExcessShare are reported separately; no composite GateScore is used.",
        "",
        "## Month-level results",
        "",
        months.to_markdown(index=False),
        "",
        "## Paired day-block bootstrap",
        "",
        summary.to_markdown(index=False),
        "",
        "Interpretation: directionally consistent gains in both pressure months, with 95% day-block intervals crossing zero.",
    ]
    (OUT / "pressure_gate_evidence_audit.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
