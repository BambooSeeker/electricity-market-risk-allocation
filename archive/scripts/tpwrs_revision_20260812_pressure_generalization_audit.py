from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "taskD_nyiso_method_transfer_20260610" / "nyiso_transfer_dataset.csv"
TOTALS = (
    ROOT
    / "outputs"
    / "tpwrs_revision_20260811_nyiso_pressure_bootstrap"
    / "nyiso_pressure_zone_month_totals.csv"
)
OUT = ROOT / "outputs" / "tpwrs_revision_20260812_pressure_generalization_audit"

VALIDATION_START = pd.Timestamp("2025-04-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
N_BOOT = 10_000
SEED = 20260812


def metrics(blocks: pd.DataFrame, method: str) -> tuple[float, float]:
    recall = blocks[f"{method}_events"].sum() / blocks["events"].sum()
    excess_share = blocks[f"{method}_excess"].sum() / blocks["excess"].sum()
    return float(recall), float(excess_share)


def zone_month_bootstrap(blocks: pd.DataFrame) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sampled = blocks.iloc[rng.integers(0, len(blocks), len(blocks))]
        local_r, local_es = metrics(sampled, "historical")
        base_r, base_es = metrics(sampled, "base")
        draws.append((local_r - base_r, local_es - base_es))
    arr = np.asarray(draws)
    return tuple(np.quantile(arr, [0.025, 0.975], axis=0).T.ravel())


def summarize(scope: str, blocks: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    base_r, base_es = metrics(blocks, "base")
    local_r, local_es = metrics(blocks, "historical")
    r_lo, r_hi, es_lo, es_hi = zone_month_bootstrap(blocks)
    monthly = []
    for month, group in blocks.groupby("month", sort=True):
        b_r, b_es = metrics(group, "base")
        l_r, l_es = metrics(group, "historical")
        monthly.append(
            {
                "scope": scope,
                "month": month,
                "zone_months": len(group),
                "delta_recall": l_r - b_r,
                "delta_excess_share": l_es - b_es,
            }
        )
    monthly_df = pd.DataFrame(monthly)

    lomo = []
    for month in sorted(blocks["month"].unique()):
        retained = blocks[blocks["month"] != month]
        if retained.empty:
            continue
        b_r, b_es = metrics(retained, "base")
        l_r, l_es = metrics(retained, "historical")
        lomo.append((l_r - b_r, l_es - b_es))
    lomo_arr = np.asarray(lomo)
    summary = {
        "scope": scope,
        "zone_months": len(blocks),
        "calendar_months": blocks["month"].nunique(),
        "base_recall": base_r,
        "local_recall": local_r,
        "delta_recall": local_r - base_r,
        "zone_month_delta_recall_lo": r_lo,
        "zone_month_delta_recall_hi": r_hi,
        "base_excess_share": base_es,
        "local_excess_share": local_es,
        "delta_excess_share": local_es - base_es,
        "zone_month_delta_excess_share_lo": es_lo,
        "zone_month_delta_excess_share_hi": es_hi,
        "min_lomo_delta_recall": float(lomo_arr[:, 0].min()),
        "min_lomo_delta_excess_share": float(lomo_arr[:, 1].min()),
        "all_lomo_positive": bool((lomo_arr > 0).all()),
    }
    return summary, monthly_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    validation = df[df["timestamp"].between(VALIDATION_START, VALIDATION_END)].copy()
    confirmation = df[df["timestamp"] > VALIDATION_END].copy()

    validation_means = (
        validation.groupby(["zone", "zone_month"], as_index=False)["load_forecast"]
        .mean()
        .rename(columns={"load_forecast": "validation_zone_month_mean"})
    )
    thresholds = (
        validation_means.groupby("zone", as_index=False)["validation_zone_month_mean"]
        .quantile(0.60)
        .rename(columns={"validation_zone_month_mean": "validation_q60_month_mean"})
    )
    confirmation_means = (
        confirmation.groupby(["zone", "zone_month", "month"], as_index=False)["load_forecast"]
        .mean()
        .rename(columns={"load_forecast": "confirmation_zone_month_mean"})
        .merge(thresholds, on="zone", how="left", validate="many_to_one")
    )
    confirmation_means["validation_referenced_load_stress"] = (
        confirmation_means["confirmation_zone_month_mean"]
        >= confirmation_means["validation_q60_month_mean"]
    )
    confirmation_means.to_csv(OUT / "validation_referenced_load_stress_flags.csv", index=False)
    thresholds.to_csv(OUT / "validation_load_thresholds.csv", index=False)

    totals = pd.read_csv(TOTALS)
    totals = totals.merge(
        confirmation_means[["zone_month", "validation_referenced_load_stress"]],
        on="zone_month",
        how="left",
        validate="one_to_one",
    )
    scopes = {
        "realized_pressure_3of4": totals[totals["realized_pressure_3of4"]],
        "realized_pressure_4of4": totals[totals["realized_pressure_4of4"]],
        "validation_referenced_load_stress": totals[
            totals["validation_referenced_load_stress"]
        ],
    }
    summaries = []
    monthly = []
    for scope, blocks in scopes.items():
        summary, monthly_df = summarize(scope, blocks)
        summaries.append(summary)
        monthly.append(monthly_df)

    summary_df = pd.DataFrame(summaries)
    monthly_df = pd.concat(monthly, ignore_index=True)
    summary_df.to_csv(OUT / "pressure_generalization_summary.csv", index=False)
    monthly_df.to_csv(OUT / "pressure_monthly_deltas.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nMonthly deltas\n", monthly_df.to_string(index=False))


if __name__ == "__main__":
    main()
