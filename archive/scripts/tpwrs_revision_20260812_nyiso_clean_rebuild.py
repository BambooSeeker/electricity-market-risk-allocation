from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).resolve().parents[1]
RAW = LEGACY / "taskA_external_market_validation" / "external_market_raw"
SPREAD = (
    LEGACY
    / "taskA_external_market_validation"
    / "external_market_processed"
    / "external_da_rt_spread_multizone.csv"
)
OLD_DATA = ROOT / "outputs" / "taskD_nyiso_method_transfer_20260610" / "nyiso_transfer_dataset.csv"
SOURCE_SCORE = (
    ROOT
    / "outputs"
    / "cross_market_direct_transfer_20260615"
    / "nyiso_direct_common_feature_scores.csv"
)
OUT = ROOT / "outputs" / "tpwrs_revision_20260812_nyiso_clean_rebuild"

BUDGET = 0.30
CORE_SHARE = 0.225
CANDIDATE_POOL = 0.50
REPLACE_FRAC = 0.05
LOWER_Q = 0.10
BURN_IN_END = pd.Timestamp("2025-03-31 23:59:59")
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
N_BOOT = 10_000
SEED = 20260812

LOAD_ZONE_MAP = {
    "CAPITL": "Capitl",
    "CENTRL": "Centrl",
    "DUNWOD": "Dunwod",
    "GENESE": "Genese",
    "HUD VL": "Hud Vl",
    "LONGIL": "Longil",
    "MHK VL": "Mhk Vl",
    "MILLWD": "Millwd",
    "N.Y.C.": "N.Y.C.",
    "NORTH": "North",
    "WEST": "West",
}


def read_day_ahead_load_vintage() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("nyiso_isolf_2025*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                issue_date = pd.to_datetime(Path(name).stem[:8], format="%Y%m%d")
                with archive.open(name) as handle:
                    frame = pd.read_csv(handle)
                frame["timestamp"] = pd.to_datetime(frame["Time Stamp"])
                frame["issue_date"] = issue_date
                frame["dst_occurrence"] = frame.groupby("timestamp").cumcount()
                frames.append(frame)

    wide = pd.concat(frames, ignore_index=True)
    wide["target_date"] = wide["timestamp"].dt.normalize()
    wide["lead_days"] = (wide["target_date"] - wide["issue_date"]).dt.days
    wide = wide[wide["lead_days"] >= 1].copy()
    wide = wide.sort_values(["timestamp", "dst_occurrence", "issue_date"])
    wide = wide.drop_duplicates(["timestamp", "dst_occurrence"], keep="last")

    records = []
    for zone, column in LOAD_ZONE_MAP.items():
        local = wide[["timestamp", "dst_occurrence", "issue_date", column]].copy()
        local["zone"] = zone
        local = local.rename(columns={column: "load_forecast"})
        records.append(local)
    return pd.concat(records, ignore_index=True)


def month_rank(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("zone_month", group_keys=False)[column].rank(method="average", pct=True)


def build_dataset(load: pd.DataFrame) -> pd.DataFrame:
    spread = pd.read_csv(SPREAD, parse_dates=["timestamp"])
    spread = spread[spread["zone"].isin(LOAD_ZONE_MAP)].copy()
    spread["dst_occurrence"] = spread.groupby(["zone", "timestamp"]).cumcount()
    df = spread.merge(load, on=["timestamp", "zone", "dst_occurrence"], how="inner", validate="one_to_one")
    df = df.sort_values(["zone", "timestamp", "dst_occurrence"]).reset_index(drop=True)

    thresholds = (
        df[df["timestamp"] <= BURN_IN_END]
        .groupby("zone")["da_rt_spread"]
        .quantile(LOWER_Q)
        .rename("tail_threshold")
    )
    df = df.merge(thresholds, on="zone", how="left", validate="many_to_one")
    df["lower_tail_event"] = (df["da_rt_spread"] <= df["tail_threshold"]).astype(int)
    df["negative_excess"] = np.maximum(df["tail_threshold"] - df["da_rt_spread"], 0.0)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["zone_month"] = df["zone"] + "_" + df["month"]

    shifted_spread = df.groupby(["zone", "hour"])["da_rt_spread"].shift(1)
    shifted_event = df.groupby(["zone", "hour"])["lower_tail_event"].shift(1)
    df["hist_q10_60"] = (
        shifted_spread.groupby([df["zone"], df["hour"]])
        .rolling(60, min_periods=10)
        .quantile(0.10)
        .reset_index(level=[0, 1], drop=True)
    )
    df["hist_tail_rate_30"] = (
        shifted_event.groupby([df["zone"], df["hour"]])
        .rolling(30, min_periods=10)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["base_score_raw"] = -df["hist_q10_60"]
    df["base_score"] = month_rank(df, "base_score_raw").fillna(0.5)
    df["historical_rank"] = month_rank(df, "hist_tail_rate_30").fillna(0.5)
    df["congestion_rank"] = month_rank(df.assign(_cong=df["day_ahead_congestion"].abs()), "_cong").fillna(0.5)
    df["sequence_score"] = df["base_score"] + 0.20 * df["historical_rank"]
    df["congestion_score"] = df["base_score"] + 0.20 * df["congestion_rank"]
    return df


def monthly_topk(df: pd.DataFrame, score: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        idx = group.sort_values([score, "timestamp"], ascending=[False, True]).head(k).index
        alert.loc[idx] = True
    return alert


def boundary_preserved(df: pd.DataFrame, base: str, fill: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        n_core = min(math.floor(len(group) * CORE_SHARE), k)
        order = group.sort_values([base, "timestamp"], ascending=[False, True])
        core = list(order.head(n_core).index)
        local = group.copy()
        local["base_rank_pct"] = local[base].rank(method="first", ascending=False, pct=True)
        candidates = local.drop(index=core)
        candidates = candidates[candidates["base_rank_pct"] <= CANDIDATE_POOL]
        fill_idx = list(candidates.sort_values([fill, "timestamp"], ascending=[False, True]).head(k - n_core).index)
        alert.loc[core + fill_idx] = True
    return alert


def load_gate(df: pd.DataFrame, start: pd.Series) -> pd.Series:
    validation = df[(df["timestamp"] > BURN_IN_END) & (df["timestamp"] <= VALIDATION_END)]
    thresholds = validation.groupby("zone")["load_forecast"].quantile(0.60)
    alert = start.copy()
    for _, group in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        cap = max(1, math.floor(k * REPLACE_FRAC))
        zone = group["zone"].iloc[0]
        local = group.copy()
        local["base_rank_pct"] = local["base_score"].rank(method="first", ascending=False, pct=True)
        current = local[alert.loc[local.index].to_numpy()]
        removable = current[current["base_rank_pct"] > CORE_SHARE]
        drop_idx = list(removable.nsmallest(cap, "base_score").index)
        candidates = local[~alert.loc[local.index].to_numpy()]
        candidates = candidates[candidates["base_rank_pct"] <= CANDIDATE_POOL]
        candidates = candidates[candidates["load_forecast"] >= thresholds[zone]]
        add_idx = list(candidates.nlargest(len(drop_idx), "sequence_score").index)
        count = min(len(drop_idx), len(add_idx))
        alert.loc[drop_idx[:count]] = False
        alert.loc[add_idx[:count]] = True
    return alert


def attach_source_score(df: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_DATA, parse_dates=["timestamp"])
    old = old[old["timestamp"] > VALIDATION_END].reset_index(drop=True)
    source = pd.read_csv(SOURCE_SCORE, parse_dates=["timestamp"])
    if len(old) != len(source):
        raise ValueError("Source-score rows do not align with the legacy confirmation dataset")
    old["zj_source_score"] = source["zj_direct_hgb_prob"].to_numpy()

    keys = df[df["timestamp"] > VALIDATION_END][["timestamp", "zone", "load_forecast"]].copy()
    matched = keys.merge(old, on=["timestamp", "zone", "load_forecast"], how="left", suffixes=("", "_old"))
    matched = matched.sort_values(["timestamp", "zone"]).drop_duplicates(["timestamp", "zone"], keep="first")
    score_map = matched.set_index(["timestamp", "zone"])["zj_source_score"]
    idx = pd.MultiIndex.from_frame(df[["timestamp", "zone"]])
    df["zj_source_score"] = score_map.reindex(idx).to_numpy()
    return df


def pressure_profile(confirmation: pd.DataFrame) -> pd.DataFrame:
    def tail_congestion(group: pd.DataFrame) -> float:
        events = group[group["lower_tail_event"].eq(1)]
        return 0.0 if events.empty else float(events["day_ahead_congestion"].abs().mean())

    profile = (
        confirmation.groupby("zone_month")
        .apply(
            lambda g: pd.Series(
                {
                    "zone": g["zone"].iloc[0],
                    "month": g["month"].iloc[0],
                    "n": len(g),
                    "event_rate": g["lower_tail_event"].mean(),
                    "negative_excess_intensity": g["negative_excess"].sum() / len(g),
                    "spread_volatility": g["da_rt_spread"].std(),
                    "congestion_tail_abs_mean": tail_congestion(g),
                    "load_forecast_mean": g["load_forecast"].mean(),
                }
            )
        )
        .reset_index()
    )
    dimensions = ["event_rate", "negative_excess_intensity", "spread_volatility", "congestion_tail_abs_mean"]
    for dimension in dimensions:
        profile[f"high_{dimension}"] = profile[dimension] >= profile[dimension].median()
    profile["pressure_score"] = profile[[f"high_{d}" for d in dimensions]].sum(axis=1)
    profile["pressure_3of4"] = profile["pressure_score"] >= 3
    profile["pressure_4of4"] = profile["pressure_score"] >= 4
    return profile


def totals(df: pd.DataFrame, alerts: dict[str, pd.Series], profile: pd.DataFrame) -> pd.DataFrame:
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
        for name, alert in alerts.items():
            selected = group[alert.loc[group.index].to_numpy()]
            row[f"{name}_events"] = selected["lower_tail_event"].sum()
            row[f"{name}_excess"] = selected["negative_excess"].sum()
            row[f"{name}_alerts"] = len(selected)
        rows.append(row)
    return pd.DataFrame(rows).merge(
        profile[["zone_month", "pressure_3of4", "pressure_4of4"]], on="zone_month", validate="one_to_one"
    )


def measures(blocks: pd.DataFrame, method: str) -> tuple[float, float]:
    return (
        float(blocks[f"{method}_events"].sum() / blocks["events"].sum()),
        float(blocks[f"{method}_excess"].sum() / blocks["excess"].sum()),
    )


def summarize_scope(name: str, blocks: pd.DataFrame, base: str, local: str) -> dict[str, object]:
    base_r, base_es = measures(blocks, base)
    local_r, local_es = measures(blocks, local)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sampled = blocks.iloc[rng.integers(0, len(blocks), len(blocks))]
        br, be = measures(sampled, base)
        lr, le = measures(sampled, local)
        draws.append((lr - br, le - be))
    intervals = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
    lomo = []
    for month in sorted(blocks["month"].unique()):
        retained = blocks[blocks["month"] != month]
        br, be = measures(retained, base)
        lr, le = measures(retained, local)
        lomo.append((lr - br, le - be))
    lomo = np.asarray(lomo)
    return {
        "scope": name,
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
    load = read_day_ahead_load_vintage()
    df = attach_source_score(build_dataset(load))
    confirmation = df[df["timestamp"] > VALIDATION_END].copy()

    base = monthly_topk(df, "base_score")
    local = boundary_preserved(df, "base_score", "sequence_score")
    congestion = boundary_preserved(df, "base_score", "congestion_score")
    gated = load_gate(df, congestion)
    source_base = monthly_topk(df, "zj_source_score")
    source_fill = month_rank(df, "zj_source_score").fillna(0.5) + 0.20 * df["historical_rank"]
    df["source_local_score"] = source_fill
    source_local = boundary_preserved(df, "zj_source_score", "source_local_score")

    alerts = {
        "base": base,
        "local": local,
        "congestion": congestion,
        "load_gate": gated,
        "source_base": source_base,
        "source_local": source_local,
    }
    profile = pressure_profile(confirmation)

    validation = df[(df["timestamp"] > BURN_IN_END) & (df["timestamp"] <= VALIDATION_END)]
    validation_month_mean = validation.groupby(["zone", "zone_month"])["load_forecast"].mean().reset_index()
    load_threshold = validation_month_mean.groupby("zone")["load_forecast"].quantile(0.60)
    confirmation_month_mean = confirmation.groupby(["zone", "zone_month"])["load_forecast"].mean().reset_index()
    confirmation_month_mean["validation_load_stress"] = confirmation_month_mean.apply(
        lambda row: row["load_forecast"] >= load_threshold[row["zone"]], axis=1
    )

    block_totals = totals(df, alerts, profile).merge(
        confirmation_month_mean[["zone_month", "validation_load_stress"]], on="zone_month", validate="one_to_one"
    )
    scopes = {
        "all_confirmation": block_totals,
        "pressure_3of4": block_totals[block_totals["pressure_3of4"]],
        "pressure_4of4": block_totals[block_totals["pressure_4of4"]],
        "validation_load_stress": block_totals[block_totals["validation_load_stress"]],
    }
    summaries = []
    for scope, blocks in scopes.items():
        summaries.append(summarize_scope(scope, blocks, "base", "local"))
    for base_name, local_name, label in [
        ("congestion", "load_gate", "load_gate_vs_congestion"),
        ("source_base", "source_local", "source_local_vs_source_base"),
    ]:
        summaries.append(summarize_scope(label, block_totals, base_name, local_name))

    dataset_columns = [
        "timestamp", "zone", "zone_month", "issue_date", "load_forecast", "da_rt_spread",
        "day_ahead_price", "day_ahead_congestion", "tail_threshold", "lower_tail_event",
        "negative_excess", "base_score", "historical_rank", "sequence_score", "zj_source_score",
    ]
    df[dataset_columns].to_csv(OUT / "nyiso_clean_dataset.csv", index=False)
    profile.to_csv(OUT / "nyiso_pressure_profile.csv", index=False)
    block_totals.to_csv(OUT / "nyiso_zone_month_totals.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "nyiso_clean_summary.csv", index=False)
    print(f"Full rows: {len(df):,}; confirmation rows: {len(confirmation):,}; zone-months: {len(block_totals)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
