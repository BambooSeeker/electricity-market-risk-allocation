from __future__ import annotations

import math
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_A = ROOT / "taskA_external_market_validation"
RAW = TASK_A / "external_market_raw"
PROCESSED = TASK_A / "external_market_processed"
OUT = ROOT / "taskD_nyiso_method_transfer_20260610"
OUT.mkdir(exist_ok=True)

YEAR = 2025
BUDGET = 0.30
LOWER_Q = 0.10
BURN_IN_END = pd.Timestamp("2025-03-31 23:59:59")
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
CORE_MARGIN = 0.075
CANDIDATE_POOL = 0.50
REPLACE_FRAC = 0.05
STRESS_Q = 0.60


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


def month_start(month: int) -> str:
    return f"{YEAR}{month:02d}01"


def download_isolf_month(month: int) -> Path:
    out = RAW / f"nyiso_isolf_{YEAR}{month:02d}.zip"
    url = f"http://mis.nyiso.com/public/csv/isolf/{month_start(month)}isolf_csv.zip"
    if not out.exists() or out.stat().st_size == 0:
        print(f"Downloading {url}")
        urlretrieve(url, out)
    return out


def read_isolf() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in range(1, 13):
        path = download_isolf_month(month)
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    frames.append(pd.read_csv(f))
    wide = pd.concat(frames, ignore_index=True)
    wide["timestamp"] = pd.to_datetime(wide["Time Stamp"])
    records = []
    for zone, load_col in LOAD_ZONE_MAP.items():
        if load_col not in wide.columns:
            continue
        tmp = wide[["timestamp", load_col]].copy()
        tmp["zone"] = zone
        tmp = tmp.rename(columns={load_col: "load_forecast"})
        records.append(tmp)
    long = pd.concat(records, ignore_index=True)
    return long.sort_values(["zone", "timestamp"]).reset_index(drop=True)


def rank_pct_by_group(df: pd.DataFrame, group_cols: list[str], col: str, ascending: bool = False) -> pd.Series:
    return df.groupby(group_cols, group_keys=False)[col].rank(method="first", ascending=ascending, pct=True)


def build_dataset() -> pd.DataFrame:
    spread_path = PROCESSED / "external_da_rt_spread_multizone.csv"
    spread = pd.read_csv(spread_path, parse_dates=["timestamp"])
    spread = spread[spread["zone"].isin(LOAD_ZONE_MAP)].copy()
    load = read_isolf()
    df = spread.merge(load, on=["timestamp", "zone"], how="inner")
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    thresholds = (
        df[df["timestamp"] <= BURN_IN_END]
        .groupby("zone")["da_rt_spread"]
        .quantile(LOWER_Q)
        .rename("tail_threshold")
    )
    df = df.drop(columns=[c for c in ["lower_tail_event", "negative_excess"] if c in df.columns])
    df = df.merge(thresholds, on="zone", how="left")
    df["lower_tail_event"] = (df["da_rt_spread"] <= df["tail_threshold"]).astype(int)
    df["negative_excess"] = np.maximum(df["tail_threshold"] - df["da_rt_spread"], 0.0)

    df["zone_month"] = df["zone"] + "_" + df["timestamp"].dt.to_period("M").astype(str)
    df["hour"] = df["timestamp"].dt.hour
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
    df["base_score"] = rank_pct_by_group(df, ["zone"], "base_score_raw", ascending=True).fillna(0.5)
    df["congestion_score"] = df.groupby("zone")["day_ahead_congestion"].transform(
        lambda s: s.abs().rank(method="first", pct=True)
    )
    df["sequence_score"] = (
        df["base_score"]
        + 0.20 * df.groupby("zone")["hist_tail_rate_30"].transform(lambda s: s.rank(method="first", pct=True)).fillna(0.5)
    )
    df["boundary_score"] = df["base_score"] + 0.20 * df["congestion_score"].fillna(0.5)
    return df


def empty_alert(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def monthly_topk(df: pd.DataFrame, score_col: str, budget: float = BUDGET) -> pd.Series:
    alert = empty_alert(df)
    for _, g in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(g) * budget))
        idx = g.sort_values(score_col, ascending=False).head(k).index
        alert.loc[idx] = True
    return alert


def boundary_preserved(df: pd.DataFrame, fill_score: str, budget: float = BUDGET, pool: float = BUDGET + 0.05) -> pd.Series:
    alert = empty_alert(df)
    for _, g in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(g) * budget))
        core_pct = max(0.0, budget - CORE_MARGIN)
        n_core = min(int(math.floor(len(g) * core_pct)), k)
        order = g.sort_values("base_score", ascending=False)
        core = list(order.head(n_core).index)
        remain = k - len(core)
        if remain <= 0:
            alert.loc[core] = True
            continue
        candidate_upper = min(1.0, pool)
        local = g.copy()
        local["base_rank_pct"] = local["base_score"].rank(method="first", ascending=False, pct=True)
        cand = local.drop(index=core)
        cand = cand[cand["base_rank_pct"] <= candidate_upper]
        if len(cand) < remain:
            cand = local.drop(index=core)
        fill = list(cand.sort_values(fill_score, ascending=False).head(remain).index)
        alert.loc[core + fill] = True
    return alert


def stress_gated(df: pd.DataFrame, start_alert: pd.Series) -> pd.Series:
    alert = start_alert.copy()
    validation = df[(df["timestamp"] > BURN_IN_END) & (df["timestamp"] <= VALIDATION_END)]
    stress_threshold = validation.groupby("zone")["load_forecast"].quantile(STRESS_Q).to_dict()
    for _, g in df.groupby("zone_month", sort=True):
        k = max(1, math.ceil(len(g) * BUDGET))
        cap = max(1, math.floor(k * REPLACE_FRAC))
        zone = str(g["zone"].iloc[0])
        threshold = stress_threshold.get(zone, np.nan)
        if not np.isfinite(threshold):
            continue
        local = g.copy()
        local["base_rank_pct"] = local["base_score"].rank(method="first", ascending=False, pct=True)
        current = local[alert.loc[local.index].to_numpy()]
        removable = current[current["base_rank_pct"] > max(0.0, BUDGET - CORE_MARGIN)]
        drop_idx = list(removable.sort_values("base_score", ascending=True).head(cap).index)
        if not drop_idx:
            continue
        candidates = local[~alert.loc[local.index].to_numpy()].copy()
        candidates = candidates[candidates["base_rank_pct"] <= CANDIDATE_POOL]
        candidates = candidates[candidates["load_forecast"] >= threshold]
        add_idx = list(candidates.sort_values("sequence_score", ascending=False).head(len(drop_idx)).index)
        if not add_idx:
            continue
        alert.loc[drop_idx[: len(add_idx)]] = False
        alert.loc[add_idx] = True
    return alert


def evaluate(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    eval_df = df[df["timestamp"] > VALIDATION_END].copy()
    rows = []
    scopes = [("all_confirmation_zone_months", eval_df)]
    scopes.extend([(f"zone_{z}", g) for z, g in eval_df.groupby("zone")])
    for scope, part in scopes:
        for method, mask in alerts.items():
            local_mask = mask.loc[part.index]
            selected = part[local_mask]
            events = int(part["lower_tail_event"].sum())
            alerts_n = int(local_mask.sum())
            selected_events = int(selected["lower_tail_event"].sum())
            total_excess = float(part["negative_excess"].sum())
            selected_excess = float(selected["negative_excess"].sum())
            precision = selected_events / alerts_n if alerts_n else np.nan
            recall = selected_events / events if events else np.nan
            excess_share = selected_excess / total_excess if total_excess > 0 else np.nan
            false_alert_burden = 1 - precision
            gate_score = recall + excess_share - 0.2 * false_alert_burden
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "n": len(part),
                    "events": events,
                    "alerts": alerts_n,
                    "precision": precision,
                    "recall": recall,
                    "excess_share": excess_share,
                    "false_alert_burden": false_alert_burden,
                    "gate_score": gate_score,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    df.to_csv(OUT / "nyiso_transfer_dataset.csv", index=False, encoding="utf-8-sig")
    alerts = {
        "Base historical lower-tail ranking": monthly_topk(df, "base_score"),
        "DA-congestion boundary ranking": boundary_preserved(df, "boundary_score", pool=BUDGET + 0.05),
        "Historical-tail-rate boundary ranking": boundary_preserved(df, "sequence_score", pool=CANDIDATE_POOL),
    }
    alerts["Load-forecast-gated boundary replacement"] = stress_gated(
        df, alerts["DA-congestion boundary ranking"]
    )
    metrics = evaluate(df, alerts)
    metrics.to_csv(OUT / "nyiso_method_transfer_metrics.csv", index=False, encoding="utf-8-sig")

    pooled = metrics[metrics["scope"].eq("all_confirmation_zone_months")].sort_values("gate_score", ascending=False)
    zone_best = (
        metrics[metrics["scope"].str.startswith("zone_")]
        .sort_values(["scope", "gate_score"], ascending=[True, False])
        .groupby("scope", as_index=False)
        .first()
    )
    lines = [
        "# NYISO Method-Transfer Experiment",
        "",
        "This experiment applies the fixed-budget boundary-preserved and stress-gated screening procedure to NYISO with NYISO-specific ex-ante variables.",
        "",
        "## Official data used",
        "",
        "- Day-ahead zonal LBMP: `http://mis.nyiso.com/public/csv/damlbmp/YYYYMM01damlbmp_zone_csv.zip`.",
        "- Real-time zonal LBMP: `http://mis.nyiso.com/public/csv/rtlbmp/YYYYMM01rtlbmp_zone_csv.zip`.",
        "- NYISO load forecast: `http://mis.nyiso.com/public/csv/isolf/YYYYMM01isolf_csv.zip`.",
        "",
        "## Protocol",
        "",
        f"- Burn-in for lower-tail threshold and history: through {BURN_IN_END.date()}.",
        f"- Validation window for the load-forecast stress threshold: through {VALIDATION_END.date()}.",
        "- Confirmation window: July to December 2025.",
        "- Ranking unit: zone-month.",
        "- Budget: top30 fixed alert set.",
        "",
        "## Pooled confirmation result",
        "",
        pooled.to_markdown(index=False),
        "",
        "## Per-zone best method summary",
        "",
        zone_best.to_markdown(index=False),
    ]
    (OUT / "nyiso_method_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
