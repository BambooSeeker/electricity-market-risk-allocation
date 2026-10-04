from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "taskD_nyiso_method_transfer_20260610" / "nyiso_transfer_dataset.csv"
PROFILE = (
    ROOT
    / "outputs"
    / "section5_rebuild_20260611"
    / "nyiso_pressure_slice_audit_20260615"
    / "nyiso_zone_month_pressure_profile.csv"
)
OUT = ROOT / "outputs" / "tpwrs_revision_20260811_nyiso_pressure_bootstrap"

BUDGET = 0.30
CORE_MARGIN = 0.075
CANDIDATE_POOL = 0.50
REPLACE_FRAC = 0.05
STRESS_Q = 0.60
BURN_IN_END = pd.Timestamp("2025-03-31 23:59:59")
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
N_BOOT = 10_000
SEED = 20260811


def monthly_topk(df: pd.DataFrame, score_col: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        alert.loc[group.nlargest(k, score_col).index] = True
    return alert


def boundary_preserved(df: pd.DataFrame, fill_score: str, pool: float) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        n_core = min(math.floor(len(group) * (BUDGET - CORE_MARGIN)), k)
        order = group.sort_values("base_score", ascending=False)
        core = list(order.head(n_core).index)
        local = group.copy()
        local["base_rank_pct"] = local["base_score"].rank(
            method="first", ascending=False, pct=True
        )
        candidates = local.drop(index=core)
        candidates = candidates[candidates["base_rank_pct"] <= pool]
        if len(candidates) < k - len(core):
            candidates = local.drop(index=core)
        fill = list(candidates.nlargest(k - len(core), fill_score).index)
        alert.loc[core + fill] = True
    return alert


def stress_gated(df: pd.DataFrame, start_alert: pd.Series) -> pd.Series:
    alert = start_alert.copy()
    validation = df[
        (df["timestamp"] > BURN_IN_END) & (df["timestamp"] <= VALIDATION_END)
    ]
    thresholds = validation.groupby("zone")["load_forecast"].quantile(STRESS_Q)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        cap = max(1, math.floor(k * REPLACE_FRAC))
        zone = group["zone"].iloc[0]
        local = group.copy()
        local["base_rank_pct"] = local["base_score"].rank(
            method="first", ascending=False, pct=True
        )
        current = local[alert.loc[local.index].to_numpy()]
        removable = current[
            current["base_rank_pct"] > BUDGET - CORE_MARGIN
        ].nsmallest(cap, "base_score")
        candidates = local[~alert.loc[local.index].to_numpy()]
        candidates = candidates[candidates["base_rank_pct"] <= CANDIDATE_POOL]
        candidates = candidates[candidates["load_forecast"] >= thresholds.loc[zone]]
        additions = candidates.nlargest(len(removable), "sequence_score")
        count = min(len(removable), len(additions))
        alert.loc[removable.head(count).index] = False
        alert.loc[additions.head(count).index] = True
    return alert


def block_totals(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for zone_month, group in df.groupby("zone_month", sort=True):
        row = {
            "zone_month": zone_month,
            "month": str(group["month"].iloc[0]),
            "events": float(group["lower_tail_event"].sum()),
            "excess": float(group["negative_excess"].sum()),
        }
        for name, mask in alerts.items():
            selected = group[mask.loc[group.index].to_numpy()]
            row[f"{name}_events"] = float(selected["lower_tail_event"].sum())
            row[f"{name}_excess"] = float(selected["negative_excess"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(blocks: pd.DataFrame, method: str) -> tuple[float, float]:
    recall = blocks[f"{method}_events"].sum() / blocks["events"].sum()
    excess_share = blocks[f"{method}_excess"].sum() / blocks["excess"].sum()
    return float(recall), float(excess_share)


def bootstrap(
    blocks: pd.DataFrame, method: str, reference: str, rng: np.random.Generator
) -> tuple[dict[str, float], pd.DataFrame]:
    n = len(blocks)
    samples = []
    for draw in range(N_BOOT):
        sampled = blocks.iloc[rng.integers(0, n, n)]
        method_r, method_es = metrics(sampled, method)
        reference_r, reference_es = metrics(sampled, reference)
        samples.append(
            {
                "draw": draw,
                "delta_recall": method_r - reference_r,
                "delta_excess_share": method_es - reference_es,
            }
        )
    sample_df = pd.DataFrame(samples)
    point_r, point_es = metrics(blocks, method)
    ref_r, ref_es = metrics(blocks, reference)
    summary = {
        "blocks": n,
        "method_recall": point_r,
        "reference_recall": ref_r,
        "delta_recall": point_r - ref_r,
        "delta_recall_lo": sample_df["delta_recall"].quantile(0.025),
        "delta_recall_hi": sample_df["delta_recall"].quantile(0.975),
        "method_excess_share": point_es,
        "reference_excess_share": ref_es,
        "delta_excess_share": point_es - ref_es,
        "delta_excess_share_lo": sample_df["delta_excess_share"].quantile(0.025),
        "delta_excess_share_hi": sample_df["delta_excess_share"].quantile(0.975),
    }
    return summary, sample_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df[df["timestamp"] > VALIDATION_END].copy()
    profile = pd.read_csv(PROFILE)
    flags = profile.set_index("zone_month")
    for column in [
        "realized_pressure_3of4",
        "realized_pressure_4of4",
        "ex_ante_load_stress_top40",
    ]:
        df[column] = df["zone_month"].map(flags[column]).astype(bool)

    full = pd.read_csv(DATA, parse_dates=["timestamp"])
    base = monthly_topk(full, "base_score")
    congestion = boundary_preserved(full, "boundary_score", BUDGET + 0.05)
    historical = boundary_preserved(full, "sequence_score", CANDIDATE_POOL)
    load_gate = stress_gated(full, congestion)
    alerts = {
        "base": base,
        "congestion": congestion,
        "historical": historical,
        "load_gate": load_gate,
    }
    totals = block_totals(df, alerts)
    for column in [
        "realized_pressure_3of4",
        "realized_pressure_4of4",
        "ex_ante_load_stress_top40",
    ]:
        selected_blocks = set(df.loc[df[column], "zone_month"].unique())
        totals[column] = totals["zone_month"].isin(selected_blocks)
    totals.to_csv(OUT / "nyiso_pressure_zone_month_totals.csv", index=False)

    scopes = {
        "realized_pressure_3of4": totals[totals["realized_pressure_3of4"]],
        "realized_pressure_4of4": totals[totals["realized_pressure_4of4"]],
        "ex_ante_load_stress_top40": totals[totals["ex_ante_load_stress_top40"]],
    }
    comparisons = [
        ("historical", "base", "local_boundary_vs_base"),
        ("load_gate", "congestion", "load_gate_vs_congestion_start"),
    ]
    rng = np.random.default_rng(SEED)
    summaries = []
    all_samples = []
    for scope, blocks in scopes.items():
        for method, reference, label in comparisons:
            summary, samples = bootstrap(blocks, method, reference, rng)
            summary.update({"scope": scope, "comparison": label})
            summaries.append(summary)
            samples.insert(0, "scope", scope)
            samples.insert(1, "comparison", label)
            all_samples.append(samples)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "nyiso_pressure_bootstrap_summary.csv", index=False)
    pd.concat(all_samples, ignore_index=True).to_csv(
        OUT / "nyiso_pressure_bootstrap_samples.csv", index=False
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
