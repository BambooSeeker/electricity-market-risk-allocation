from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "taskD_nyiso_method_transfer_20260610" / "nyiso_transfer_dataset.csv"
SOURCE = ROOT / "outputs" / "cross_market_direct_transfer_20260615" / "nyiso_direct_common_feature_scores.csv"
PRESSURE = ROOT / "outputs" / "tpwrs_revision_20260811_nyiso_pressure_bootstrap" / "nyiso_pressure_zone_month_totals.csv"
LOAD_STRESS = ROOT / "outputs" / "tpwrs_revision_20260812_pressure_generalization_audit" / "validation_referenced_load_stress_flags.csv"
OUT = ROOT / "outputs" / "tpwrs_revision_20260812_nyiso_frozen_normalization"

BUDGET = 0.30
CORE_SHARE = 0.225
CANDIDATE_POOL = 0.50
VALIDATION_START = pd.Timestamp("2025-04-01")
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
N_BOOT = 10_000
SEED = 20260812


def frozen_mid_ecdf(values: pd.Series, reference: pd.Series) -> np.ndarray:
    ordered = np.sort(reference.dropna().to_numpy())
    result = np.full(len(values), 0.5, dtype=float)
    valid = values.notna().to_numpy()
    current = values.to_numpy()[valid]
    left = np.searchsorted(ordered, current, side="left")
    right = np.searchsorted(ordered, current, side="right")
    result[valid] = (left + right) / (2 * len(ordered))
    return result


def topk(df: pd.DataFrame, score: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        alert.loc[group.sort_values(score, ascending=False, kind="stable").head(k).index] = True
    return alert


def boundary(df: pd.DataFrame, base: str, fill: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        n_core = min(math.floor(len(group) * CORE_SHARE), k)
        core = list(group.sort_values(base, ascending=False, kind="stable").head(n_core).index)
        local = group.copy()
        local["base_rank_pct"] = local[base].rank(method="first", ascending=False, pct=True)
        candidates = local.drop(index=core)
        candidates = candidates[candidates["base_rank_pct"] <= CANDIDATE_POOL]
        fill_idx = list(candidates.sort_values(fill, ascending=False, kind="stable").head(k - n_core).index)
        alert.loc[core + fill_idx] = True
    return alert


def block_totals(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    confirmation = df[df["timestamp"] > VALIDATION_END]
    for zone_month, group in confirmation.groupby("zone_month", sort=True):
        row = {
            "zone_month": zone_month,
            "zone": group["zone"].iloc[0],
            "month": group["month"].iloc[0],
            "events": group["lower_tail_event"].sum(),
            "excess": group["negative_excess"].sum(),
        }
        for name, mask in alerts.items():
            selected = group[mask.loc[group.index].to_numpy()]
            row[f"{name}_events"] = selected["lower_tail_event"].sum()
            row[f"{name}_excess"] = selected["negative_excess"].sum()
            row[f"{name}_alerts"] = len(selected)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(blocks: pd.DataFrame, method: str) -> tuple[float, float]:
    return (
        float(blocks[f"{method}_events"].sum() / blocks["events"].sum()),
        float(blocks[f"{method}_excess"].sum() / blocks["excess"].sum()),
    )


def summary(scope: str, blocks: pd.DataFrame, base: str, local: str) -> dict[str, object]:
    base_r, base_es = metrics(blocks, base)
    local_r, local_es = metrics(blocks, local)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sampled = blocks.iloc[rng.integers(0, len(blocks), len(blocks))]
        br, be = metrics(sampled, base)
        lr, le = metrics(sampled, local)
        draws.append((lr - br, le - be))
    intervals = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)

    lomo = []
    for month in sorted(blocks["month"].unique()):
        retained = blocks[blocks["month"] != month]
        br, be = metrics(retained, base)
        lr, le = metrics(retained, local)
        lomo.append((lr - br, le - be))
    lomo = np.asarray(lomo)
    return {
        "scope": scope,
        "zone_months": len(blocks),
        "calendar_months": blocks["month"].nunique(),
        "base_recall": base_r,
        "local_recall": local_r,
        "delta_recall": local_r - base_r,
        "delta_recall_lo": intervals[0, 0],
        "delta_recall_hi": intervals[1, 0],
        "base_excess_share": base_es,
        "local_excess_share": local_es,
        "delta_excess_share": local_es - base_es,
        "delta_excess_share_lo": intervals[0, 1],
        "delta_excess_share_hi": intervals[1, 1],
        "min_lomo_delta_recall": lomo[:, 0].min(),
        "min_lomo_delta_excess_share": lomo[:, 1].min(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    validation = df[df["timestamp"].between(VALIDATION_START, VALIDATION_END)]
    for zone, index in df.groupby("zone").groups.items():
        reference = validation[validation["zone"] == zone]
        df.loc[index, "base_frozen"] = frozen_mid_ecdf(df.loc[index, "base_score_raw"], reference["base_score_raw"])
        df.loc[index, "history_frozen"] = frozen_mid_ecdf(df.loc[index, "hist_tail_rate_30"], reference["hist_tail_rate_30"])
    df["local_frozen"] = df["base_frozen"] + 0.20 * df["history_frozen"]

    source = pd.read_csv(SOURCE)
    confirmation = df["timestamp"] > VALIDATION_END
    df.loc[confirmation, "source_score"] = source["zj_direct_hgb_prob"].to_numpy()
    original_source_alert = source["zj_direct_alert"].astype(str).str.lower().eq("true").to_numpy()
    df.loc[confirmation, "source_alert"] = original_source_alert
    df["source_key"] = df["source_score"].fillna(-1.0) + df["source_alert"].fillna(False).astype(float) * 1e-10
    df["source_local_frozen"] = df["source_key"] + 0.20 * df["history_frozen"]

    base = topk(df, "base_frozen")
    local = boundary(df, "base_frozen", "local_frozen")
    source_base = pd.Series(False, index=df.index)
    source_base.loc[confirmation] = original_source_alert
    source_local = boundary(df, "source_key", "source_local_frozen")
    alerts = {"base": base, "local": local, "source_base": source_base, "source_local": source_local}

    blocks = block_totals(df, alerts)
    pressure = pd.read_csv(PRESSURE)[["zone_month", "realized_pressure_3of4", "realized_pressure_4of4"]]
    load = pd.read_csv(LOAD_STRESS)[["zone_month", "validation_referenced_load_stress"]]
    blocks = blocks.merge(pressure, on="zone_month", validate="one_to_one").merge(load, on="zone_month", validate="one_to_one")
    scopes = {
        "all_confirmation": blocks,
        "pressure_3of4": blocks[blocks["realized_pressure_3of4"]],
        "pressure_4of4": blocks[blocks["realized_pressure_4of4"]],
        "validation_load_stress": blocks[blocks["validation_referenced_load_stress"]],
    }
    rows = [summary(name, part, "base", "local") for name, part in scopes.items()]
    rows.append(summary("source_local_all", blocks, "source_base", "source_local"))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "nyiso_frozen_normalization_summary.csv", index=False)
    blocks.to_csv(OUT / "nyiso_frozen_normalization_zone_month_totals.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
