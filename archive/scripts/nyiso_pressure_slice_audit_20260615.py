from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASKD = ROOT / "taskD_nyiso_method_transfer_20260610"
OUT = ROOT / "section5_rebuild_20260611" / "nyiso_pressure_slice_audit_20260615"
OUT.mkdir(parents=True, exist_ok=True)

BUDGET = 0.30
VALIDATION_END = pd.Timestamp("2025-06-30 23:59:59")
BURN_IN_END = pd.Timestamp("2025-03-31 23:59:59")
CORE_MARGIN = 0.075
CANDIDATE_POOL = 0.50
REPLACE_FRAC = 0.05
STRESS_Q = 0.60


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
        n_core = min(int(math.floor(len(g) * max(0.0, budget - CORE_MARGIN))), k)
        order = g.sort_values("base_score", ascending=False)
        core = list(order.head(n_core).index)
        remain = k - len(core)
        if remain <= 0:
            alert.loc[core] = True
            continue
        local = g.copy()
        local["base_rank_pct"] = local["base_score"].rank(method="first", ascending=False, pct=True)
        cand = local.drop(index=core)
        cand = cand[cand["base_rank_pct"] <= min(1.0, pool)]
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


def evaluate(part: pd.DataFrame, alerts: dict[str, pd.Series], scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                "zone_months": int(part["zone_month"].nunique()),
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
    return rows


def build_pressure_profile(eval_df: pd.DataFrame) -> pd.DataFrame:
    def lower_tail_cong_abs(s: pd.DataFrame) -> float:
        ev = s[s["lower_tail_event"].eq(1)]
        if ev.empty:
            return 0.0
        return float(ev["day_ahead_congestion"].abs().mean())

    profile = (
        eval_df.groupby("zone_month")
        .apply(
            lambda g: pd.Series(
                {
                    "zone": g["zone"].iloc[0],
                    "month": g["month"].iloc[0],
                    "n": len(g),
                    "event_rate": g["lower_tail_event"].mean(),
                    "negative_excess_intensity": g["negative_excess"].sum() / len(g),
                    "spread_volatility": g["da_rt_spread"].std(),
                    "congestion_tail_abs_mean": lower_tail_cong_abs(g),
                    "load_forecast_mean": g["load_forecast"].mean(),
                }
            )
        )
        .reset_index()
    )
    dims = ["event_rate", "negative_excess_intensity", "spread_volatility", "congestion_tail_abs_mean"]
    for d in dims:
        threshold = profile[d].median()
        profile[f"high_{d}"] = profile[d] >= threshold
    profile["realized_pressure_score"] = profile[[f"high_{d}" for d in dims]].sum(axis=1)
    profile["realized_pressure_3of4"] = profile["realized_pressure_score"] >= 3
    profile["realized_pressure_4of4"] = profile["realized_pressure_score"] >= 4
    profile["load_stress_within_zone"] = profile.groupby("zone")["load_forecast_mean"].rank(method="first", pct=True)
    profile["ex_ante_load_stress_top40"] = profile["load_stress_within_zone"] >= 0.60
    return profile


def main() -> None:
    df = pd.read_csv(TASKD / "nyiso_transfer_dataset.csv", parse_dates=["timestamp"])
    eval_df = df[df["timestamp"] > VALIDATION_END].copy()

    alerts = {
        "Base historical lower-tail ranking": monthly_topk(df, "base_score"),
        "DA-congestion boundary ranking": boundary_preserved(df, "boundary_score", pool=BUDGET + 0.05),
        "Historical-tail-rate boundary ranking": boundary_preserved(df, "sequence_score", pool=CANDIDATE_POOL),
    }
    alerts["Load-forecast-gated boundary replacement"] = stress_gated(
        df, alerts["DA-congestion boundary ranking"]
    )

    profile = build_pressure_profile(eval_df)
    profile.to_csv(OUT / "nyiso_zone_month_pressure_profile.csv", index=False, encoding="utf-8-sig")
    # Preserve the original row index so precomputed alert masks remain aligned.
    flag_cols = [
        "realized_pressure_score",
        "realized_pressure_3of4",
        "realized_pressure_4of4",
        "ex_ante_load_stress_top40",
    ]
    flags = profile.set_index("zone_month")[flag_cols]
    for col in flag_cols:
        eval_df[col] = eval_df["zone_month"].map(flags[col])

    scopes = {
        "all_confirmation_zone_months": eval_df,
        "realized_pressure_3of4_zone_months": eval_df[eval_df["realized_pressure_3of4"]],
        "realized_non_pressure_3of4_zone_months": eval_df[~eval_df["realized_pressure_3of4"]],
        "realized_pressure_4of4_zone_months": eval_df[eval_df["realized_pressure_4of4"]],
        "ex_ante_load_stress_top40_zone_months": eval_df[eval_df["ex_ante_load_stress_top40"]],
        "ex_ante_load_nonstress_zone_months": eval_df[~eval_df["ex_ante_load_stress_top40"]],
    }
    rows = []
    for scope, part in scopes.items():
        if not part.empty:
            rows.extend(evaluate(part, alerts, scope))
    metrics = pd.DataFrame(rows)
    original = pd.read_csv(TASKD / "nyiso_method_transfer_metrics.csv")
    original_all = original[original["scope"].eq("all_confirmation_zone_months")].set_index("method")
    rebuilt_all = metrics[metrics["scope"].eq("all_confirmation_zone_months")].set_index("method")
    for method in original_all.index.intersection(rebuilt_all.index):
        old = float(original_all.loc[method, "recall"])
        new = float(rebuilt_all.loc[method, "recall"])
        if abs(old - new) > 1e-9:
            raise RuntimeError(f"All-scope recall mismatch for {method}: original={old}, rebuilt={new}")
    metrics.to_csv(OUT / "nyiso_pressure_slice_method_metrics.csv", index=False, encoding="utf-8-sig")

    best = (
        metrics.sort_values(["scope", "gate_score"], ascending=[True, False])
        .groupby("scope", as_index=False)
        .first()
    )
    best.to_csv(OUT / "nyiso_pressure_slice_best_methods.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# NYISO pressure-slice audit, 2026-06-15",
        "",
        "The audit creates two external pressure slices: a realized pressure zone-month slice based on four realized market-risk dimensions, and an ex-ante load-stress slice based only on zone-normalized load forecasts.",
        "",
        "## Zone-month counts",
        "",
        f"- Confirmation zone-months: {profile['zone_month'].nunique()}",
        f"- Realized pressure 3-of-4 zone-months: {int(profile['realized_pressure_3of4'].sum())}",
        f"- Realized pressure 4-of-4 zone-months: {int(profile['realized_pressure_4of4'].sum())}",
        f"- Ex-ante load-stress top40 zone-months: {int(profile['ex_ante_load_stress_top40'].sum())}",
        "",
        "## Best method by slice",
        "",
        best.to_markdown(index=False),
    ]
    (OUT / "nyiso_pressure_slice_audit_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
