from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from c20_distributional_tail_risk_scores import build_long_feature_panel
from c31_conformalized_deep_ensemble import add_features, feature_cols


ROOT = Path("C:/Users/MECHREVO/Documents/Codex/2026-05-09/new-chat")
OUT = ROOT / "foundation_model_mechanism_experiment_20260715"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_PATH = ROOT / "c38_modern_probabilistic_complement_audit" / "c38_modern_complement_predictions.csv"
TABPFN_PATH = OUT / "tabpfn_v2_predictions.csv"
CHRONOS_PATH = OUT / "chronos2_predictions.csv"

PRESSURE_MONTHS = {"2025-09", "2025-10"}
CONFIRMATION_MONTHS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
BUDGET = 0.30
CORE = 0.225
CANDIDATE_POOL = 0.40
REPLACEMENT_POOL = 0.50
REPLACEMENT_FRAC = 0.05
ALPHA = 0.20
SEED = 20260715

os.environ.setdefault("TABPFN_MAX_BATCHED_TEST_ROWS", "128")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TABPFN_TRAIN_LIMIT = 10_000
CHRONOS_CONTEXT = 672
CHRONOS_HORIZON = 48
CHRONOS_QUANTILES = [0.05, 0.10, 0.20, 0.50]
CHRONOS_COVARIATES = [
    "price_day_ahead_YQFDC-01",
    "load_day_ahead_pred",
    "net_load_pred",
    "renewable_pred",
    "renewable_share",
    "machine_state_sum",
    "cong_abs_max",
    "cong_range",
    "node_price_range",
    "da_cong_negative_pressure",
    "hour_sin",
    "hour_cos",
]


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    evaluation = pd.read_csv(EVAL_PATH, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    panel = add_features(build_long_feature_panel()).sort_values("time").reset_index(drop=True)
    features = feature_cols(panel)
    return evaluation, panel, features


def clean_features(
    core: pd.DataFrame,
    frames: list[pd.DataFrame],
    features: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    x_core = core[features].replace([np.inf, -np.inf], np.nan)
    usable = x_core.notna().any(axis=0)
    used = list(x_core.columns[usable])
    med = x_core[used].median(numeric_only=True).fillna(0.0)
    x_core = x_core[used].fillna(med).fillna(0.0).to_numpy(np.float32)
    arrays = [
        frame[used].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).to_numpy(np.float32)
        for frame in frames
    ]
    return x_core, arrays, used


def run_tabpfn() -> None:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    evaluation, panel, features = load_frames()
    rows: list[pd.DataFrame] = []
    audits: list[dict] = []

    for month in CONFIRMATION_MONTHS:
        test_start = pd.Timestamp(f"{month}-01")
        cal_month = (test_start - pd.offsets.MonthBegin(1)).to_period("M").strftime("%Y-%m")
        test_times = evaluation.loc[evaluation["test_month"].eq(month), "time"]
        test = panel[panel["time"].isin(test_times)].copy().sort_values("time").reset_index(drop=True)
        train_all = panel[panel["time"] < test_start].copy().sort_values("time").reset_index(drop=True)
        cal = train_all[train_all["month"].eq(cal_month)].copy().reset_index(drop=True)
        core = train_all[~train_all["month"].eq(cal_month)].copy().reset_index(drop=True)
        if len(core) > TABPFN_TRAIN_LIMIT:
            core = core.tail(TABPFN_TRAIN_LIMIT).copy().reset_index(drop=True)
        tau = float(evaluation.loc[evaluation["test_month"].eq(month), "tau_negative"].iloc[0])

        x_core, (x_cal, x_test), used = clean_features(core, [cal, test], features)
        y_core = (core["spread_rt_minus_da"].to_numpy(float) < tau).astype(np.int64)
        y_cal = (cal["spread_rt_minus_da"].to_numpy(float) < tau).astype(np.int64)

        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device="cuda",
            n_estimators=4,
            fit_mode="fit_with_cache",
            keep_cache_on_device=False,
            random_state=SEED,
            show_progress_bar=False,
        )
        model.fit(x_core, y_core)
        p_cal = model.predict_proba(x_cal)[:, 1]
        p_test_raw = model.predict_proba(x_test)[:, 1]
        p_test_iso = p_test_raw.copy()
        if len(np.unique(y_cal)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, y_cal)
            p_test_iso = iso.predict(p_test_raw)

        out = test[["time"]].copy()
        out["test_month"] = month
        out["tabpfn_v2_prob_raw"] = p_test_raw
        out["tabpfn_v2_prob_iso"] = p_test_iso
        rows.append(out)
        audits.append(
            {
                "test_month": month,
                "calibration_month": cal_month,
                "train_rows": len(core),
                "calibration_rows": len(cal),
                "test_rows": len(test),
                "features": len(used),
                "tail_threshold": tau,
                "train_event_rate": float(y_core.mean()),
                "calibration_event_rate": float(y_cal.mean()),
            }
        )
        print(f"TabPFN {month}: train={len(core)} cal={len(cal)} test={len(test)} features={len(used)}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    pd.concat(rows, ignore_index=True).to_csv(TABPFN_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(OUT / "tabpfn_v2_fold_audit.csv", index=False, encoding="utf-8-sig")


def fill_series(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    s = s.ffill().bfill()
    if s.isna().any():
        median = s.median()
        s = s.fillna(float(median) if np.isfinite(median) else fallback)
    return s.to_numpy(np.float32)


def chronos_tasks(evaluation: pd.DataFrame, panel: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    covariates = [c for c in CHRONOS_COVARIATES if c in panel.columns]
    time_to_index = pd.Series(panel.index, index=panel["time"]).to_dict()
    tasks: list[dict] = []
    meta: list[dict] = []

    for row in evaluation[["time", "test_month", "tau_negative"]].itertuples(index=False):
        idx = int(time_to_index[row.time])
        origin = idx - CHRONOS_HORIZON
        start = max(0, origin - CHRONOS_CONTEXT + 1)
        future_start = origin + 1
        if origin < 96 or idx - future_start + 1 != CHRONOS_HORIZON:
            raise ValueError(f"Insufficient Chronos history for {row.time}")
        context = panel.iloc[start : origin + 1]
        future = panel.iloc[future_start : idx + 1]
        task = {
            "target": fill_series(context["spread_rt_minus_da"].to_numpy()),
            "past_covariates": {
                c: fill_series(context[c].to_numpy()) for c in covariates
            },
            "future_covariates": {
                c: fill_series(future[c].to_numpy()) for c in covariates
            },
        }
        tasks.append(task)
        meta.append(
            {
                "time": row.time,
                "test_month": row.test_month,
                "tau_negative": row.tau_negative,
                "context_rows": len(context),
                "forecast_horizon": CHRONOS_HORIZON,
                "covariates": len(covariates),
            }
        )
    return tasks, pd.DataFrame(meta)


def run_chronos() -> None:
    from chronos import Chronos2Pipeline

    evaluation, panel, _ = load_frames()
    tasks, meta = chronos_tasks(evaluation, panel)
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    quantiles, means = pipeline.predict_quantiles(
        tasks,
        prediction_length=CHRONOS_HORIZON,
        quantile_levels=CHRONOS_QUANTILES,
        batch_size=64,
        context_length=CHRONOS_CONTEXT,
    )

    q_last = np.vstack([q.detach().float().cpu().numpy()[0, -1, :] for q in quantiles])
    mean_last = np.array([m.detach().float().cpu().numpy()[0, -1] for m in means])
    out = meta[["time", "test_month", "tau_negative"]].copy()
    for j, level in enumerate(CHRONOS_QUANTILES):
        tag = f"q{int(level * 100):02d}"
        out[f"chronos2_{tag}"] = q_last[:, j]
        out[f"chronos2_{tag}_risk"] = out["tau_negative"] - out[f"chronos2_{tag}"]
    out["chronos2_mean"] = mean_last
    downside = np.maximum(out["tau_negative"].to_numpy()[:, None] - q_last[:, :3], 0.0)
    out["chronos2_expected_shortfall_proxy"] = downside.mean(axis=1)
    out.to_csv(CHRONOS_PATH, index=False, encoding="utf-8-sig")
    meta.to_csv(OUT / "chronos2_information_audit.csv", index=False, encoding="utf-8-sig")


def rank_by_month(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("test_month")[column].rank(method="average", pct=True).fillna(0.5)


def prepare_evaluation() -> pd.DataFrame:
    df = pd.read_csv(EVAL_PATH, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    tab = pd.read_csv(TABPFN_PATH, parse_dates=["time"])[["time", "tabpfn_v2_prob_raw", "tabpfn_v2_prob_iso"]]
    chronos = pd.read_csv(CHRONOS_PATH, parse_dates=["time"])
    chronos_cols = [c for c in chronos.columns if c == "time" or c.startswith("chronos2_")]
    df = df.merge(tab, on="time", how="left").merge(chronos[chronos_cols], on="time", how="left")

    hgb_a = df["c20_blend_c20_hgb_tail_prob_cls__C13_best_exact_w0p2"].rank(method="average", pct=True).fillna(0.5)
    stable_b = df["c20_blend_c20_iso_tail_prob__C19_high_recall_pair_proxy_w0p2"].rank(
        method="average", pct=True
    ).fillna(0.5)
    df["hgb_base"] = 0.6 * hgb_a + 0.4 * stable_b
    df["tabpfn_base"] = 0.6 * rank_by_month(df, "tabpfn_v2_prob_iso") + 0.4 * stable_b
    df["boundary_bonus"] = rank_by_month(df, "da_cong_negative_pressure_delta48")
    df["qmlp_signal"] = df["modern_deep_rank"].fillna(0.5)
    chronos_parts = [
        rank_by_month(df, "chronos2_q10_risk"),
        rank_by_month(df, "chronos2_q20_risk"),
        rank_by_month(df, "chronos2_expected_shortfall_proxy"),
    ]
    df["chronos2_signal"] = pd.concat(chronos_parts, axis=1).mean(axis=1)

    for base in ["hgb_base", "tabpfn_base"]:
        for boundary in ["qmlp_signal", "chronos2_signal"]:
            df[f"fill__{base}__{boundary}"] = df[base] + ALPHA * df["boundary_bonus"] + ALPHA * df[boundary]
    return df


def topk(df: pd.DataFrame, score_col: str, budget: float = BUDGET) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("test_month", sort=True):
        k = max(1, math.ceil(len(group) * budget))
        alert.loc[group.sort_values(score_col, ascending=False).head(k).index] = True
    return alert


def protected_boundary(df: pd.DataFrame, base_col: str, fill_col: str) -> pd.Series:
    alert = pd.Series(False, index=df.index)
    for _, group in df.groupby("test_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        n_core = min(int(math.floor(len(group) * CORE)), k)
        order = group.sort_values(base_col, ascending=False)
        core_idx = list(order.head(n_core).index)
        candidates = order.iloc[n_core:].copy()
        candidates["_rank"] = candidates[base_col].rank(method="first", ascending=False, pct=True)
        candidates = candidates[candidates["_rank"] <= CANDIDATE_POOL]
        if len(candidates) < k - n_core:
            candidates = order.iloc[n_core:].copy()
        fill_idx = list(candidates.sort_values(fill_col, ascending=False).head(k - n_core).index)
        alert.loc[core_idx + fill_idx] = True
    return alert


def gated_replace(df: pd.DataFrame, base_col: str, fill_col: str, start: pd.Series) -> pd.Series:
    threshold = float(df.loc[~df["test_month"].isin(PRESSURE_MONTHS), "net_load_pred"].quantile(0.60))
    gate = df["net_load_pred"] >= threshold
    alert = start.copy()
    for _, group in df.groupby("test_month", sort=True):
        k = max(1, math.ceil(len(group) * BUDGET))
        cap = max(1, math.floor(k * REPLACEMENT_FRAC))
        base_rank = group[base_col].rank(method="first", ascending=False, pct=True)
        current = group[alert.loc[group.index].to_numpy()].copy()
        removable = current[base_rank.loc[current.index] > CORE]
        drop_idx = list(removable.sort_values(base_col, ascending=True).head(cap).index)
        candidates = group[~alert.loc[group.index].to_numpy()].copy()
        candidates = candidates[base_rank.loc[candidates.index] <= REPLACEMENT_POOL]
        candidates = candidates[gate.loc[candidates.index]]
        add_idx = list(candidates.sort_values(fill_col, ascending=False).head(len(drop_idx)).index)
        alert.loc[drop_idx[: len(add_idx)]] = False
        alert.loc[add_idx] = True
    return alert


def metric(df: pd.DataFrame, alert: pd.Series) -> dict[str, float]:
    local = alert.loc[df.index].astype(bool)
    chosen = df.loc[local]
    events = float(df["negative_tail"].sum())
    selected_events = float(chosen["negative_tail"].sum())
    alerts = int(local.sum())
    total_excess = float(df["negative_excess"].sum())
    selected_excess = float(chosen["negative_excess"].sum())
    precision = selected_events / alerts
    recall = selected_events / events
    excess_share = selected_excess / total_excess
    return {
        "alerts": alerts,
        "selected_events": selected_events,
        "recall": recall,
        "excess_share": excess_share,
        "gate_score": recall + excess_share - 0.2 * (1.0 - precision),
    }


def calibration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, column in {
        "HGB tail probability": "c20_hgb_tail_prob_cls",
        "TabPFN v2 raw": "tabpfn_v2_prob_raw",
        "TabPFN v2 isotonic": "tabpfn_v2_prob_iso",
    }.items():
        y = df["negative_tail"].to_numpy(float)
        p = np.clip(df[column].to_numpy(float), 0.0, 1.0)
        brier = float(np.mean((p - y) ** 2))
        bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
        ece = float(sum(len(g) / len(df) * abs(g["p"].mean() - g["y"].mean()) for _, g in grouped))
        rows.append({"model": model, "brier": brier, "ece_decile": ece})
    return pd.DataFrame(rows)


def foundation_weight_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    tab_rank = rank_by_month(df, "tabpfn_v2_prob_iso")
    rows: list[dict] = []
    masks: dict[str, pd.Series] = {}
    for tab_weight in [0.0, 0.05, 0.10, 0.20]:
        for chronos_weight in [0.0, 0.05, 0.10, 0.20]:
            base_col = "_sensitivity_base"
            fill_col = "_sensitivity_fill"
            df[base_col] = df["hgb_base"] + tab_weight * tab_rank
            df[fill_col] = (
                df[base_col]
                + ALPHA * df["boundary_bonus"]
                + ALPHA * df["qmlp_signal"]
                + chronos_weight * df["chronos2_signal"]
            )
            local_masks = {
                "full ranking": topk(df, fill_col),
                "core preserved": protected_boundary(df, base_col, fill_col),
            }
            for rule, mask in local_masks.items():
                key = f"tab{tab_weight:.2f}_chronos{chronos_weight:.2f}_{rule}"
                masks[key] = mask.copy()
                for scope, part in {
                    "all months": df,
                    "pressure months": df[df["test_month"].isin(PRESSURE_MONTHS)],
                }.items():
                    rows.append(
                        {
                            "tabpfn_weight": tab_weight,
                            "chronos2_weight": chronos_weight,
                            "rule": rule,
                            "scope": scope,
                            **metric(part, mask),
                        }
                    )
    return pd.DataFrame(rows), masks


def paired_month_bootstrap(
    df: pd.DataFrame,
    treatment: pd.Series,
    control: pd.Series,
    treatment_name: str,
    control_name: str,
    draws: int = 10_000,
) -> dict[str, float | str | int]:
    def components(mask: pd.Series) -> np.ndarray:
        rows = []
        for _, group in df.groupby("test_month", sort=True):
            local = mask.loc[group.index].astype(bool)
            chosen = group.loc[local]
            rows.append(
                [
                    group["negative_tail"].sum(),
                    local.sum(),
                    chosen["negative_tail"].sum(),
                    group["negative_excess"].sum(),
                    chosen["negative_excess"].sum(),
                ]
            )
        return np.asarray(rows, dtype=float)

    def scores(parts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = parts.sum(axis=-2)
        recall = total[..., 2] / total[..., 0]
        excess = total[..., 4] / total[..., 3]
        precision = total[..., 2] / total[..., 1]
        gate = recall + excess - 0.2 * (1.0 - precision)
        return recall, excess, gate

    t = components(treatment)
    c = components(control)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(t), size=(draws, len(t)))
    t_boot = scores(t[idx])
    c_boot = scores(c[idx])
    t_obs = scores(t)
    c_obs = scores(c)
    deltas = [t_boot[i] - c_boot[i] for i in range(3)]
    obs = [float(t_obs[i] - c_obs[i]) for i in range(3)]
    return {
        "treatment": treatment_name,
        "control": control_name,
        "months": len(t),
        "delta_recall": obs[0],
        "recall_ci_low": float(np.quantile(deltas[0], 0.025)),
        "recall_ci_high": float(np.quantile(deltas[0], 0.975)),
        "delta_excess_share": obs[1],
        "excess_ci_low": float(np.quantile(deltas[1], 0.025)),
        "excess_ci_high": float(np.quantile(deltas[1], 0.975)),
        "delta_gate_score": obs[2],
        "gate_ci_low": float(np.quantile(deltas[2], 0.025)),
        "gate_ci_high": float(np.quantile(deltas[2], 0.975)),
    }


def evaluate() -> None:
    df = prepare_evaluation()
    rows: list[dict] = []
    membership: list[dict] = []
    combos = [
        ("HGB", "Quantile-MLP", "hgb_base", "qmlp_signal"),
        ("TabPFN v2", "Quantile-MLP", "tabpfn_base", "qmlp_signal"),
        ("HGB", "Chronos-2", "hgb_base", "chronos2_signal"),
        ("TabPFN v2", "Chronos-2", "tabpfn_base", "chronos2_signal"),
    ]
    for base_name, boundary_name, base_col, boundary_col in combos:
        fill_col = f"fill__{base_col}__{boundary_col}"
        alerts = {
            "full ranking": topk(df, fill_col),
            "core preserved": protected_boundary(df, base_col, fill_col),
        }
        alerts["stress gated"] = gated_replace(df, base_col, fill_col, alerts["core preserved"])
        for rule, mask in alerts.items():
            for scope, part in {
                "all months": df,
                "pressure months": df[df["test_month"].isin(PRESSURE_MONTHS)],
                "non-pressure months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
            }.items():
                rows.append(
                    {
                        "base_model": base_name,
                        "boundary_model": boundary_name,
                        "rule": rule,
                        "scope": scope,
                        **metric(part, mask),
                    }
                )
        for month, group in df.groupby("test_month", sort=True):
            k = int(alerts["core preserved"].loc[group.index].sum())
            core_n = min(int(math.floor(len(group) * CORE)), k)
            core_idx = group.sort_values(base_col, ascending=False).head(core_n).index
            membership.append(
                {
                    "test_month": month,
                    "base_model": base_name,
                    "boundary_model": boundary_name,
                    "protected_core": core_n,
                    "core_retention": float(alerts["core preserved"].loc[core_idx].mean()),
                    "core_vs_full_exchanges": int(
                        (alerts["core preserved"].loc[group.index] != alerts["full ranking"].loc[group.index]).sum() // 2
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    membership_df = pd.DataFrame(membership)
    calibration = calibration_metrics(df)
    sensitivity, sensitivity_masks = foundation_weight_sensitivity(df)
    bootstrap_pairs = [
        (
            "tab0.00_chronos0.05_full ranking",
            "tab0.00_chronos0.00_full ranking",
        ),
        (
            "tab0.05_chronos0.00_core preserved",
            "tab0.00_chronos0.00_core preserved",
        ),
        (
            "tab0.05_chronos0.05_core preserved",
            "tab0.00_chronos0.00_core preserved",
        ),
    ]
    bootstrap = pd.DataFrame(
        [
            paired_month_bootstrap(
                df,
                sensitivity_masks[treatment],
                sensitivity_masks[control],
                treatment,
                control,
            )
            for treatment, control in bootstrap_pairs
        ]
    )
    metrics.to_csv(OUT / "foundation_factorial_metrics.csv", index=False, encoding="utf-8-sig")
    membership_df.to_csv(OUT / "foundation_core_membership_audit.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUT / "foundation_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT / "foundation_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT / "foundation_weight_bootstrap.csv", index=False, encoding="utf-8-sig")
    df[[
        "time",
        "test_month",
        "negative_tail",
        "negative_excess",
        "hgb_base",
        "tabpfn_base",
        "qmlp_signal",
        "chronos2_signal",
    ]].to_csv(OUT / "foundation_factorial_scores.csv", index=False, encoding="utf-8-sig")

    focus = metrics[metrics["scope"].isin(["all months", "pressure months"])].copy()
    report = [
        "# Foundation-model mechanism experiment",
        "",
        "All variants use the same confirmation months, fixed top30 budget, 22.5% protected core, candidate pool, and predicted-net-load gate.",
        "",
        "## Fixed-budget factorial metrics",
        "",
        focus.to_string(index=False),
        "",
        "## Probability calibration",
        "",
        calibration.to_string(index=False),
        "",
        "## Small-weight sensitivity",
        "",
        sensitivity[sensitivity["scope"].eq("all months")]
        .sort_values("gate_score", ascending=False)
        .head(20)
        .to_string(index=False),
        "",
        "## Paired month bootstrap",
        "",
        bootstrap.to_string(index=False),
        "",
        "## Core retention",
        "",
        membership_df.groupby(["base_model", "boundary_model"])[["core_retention", "core_vs_full_exchanges"]].mean().to_string(),
    ]
    (OUT / "foundation_model_experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(focus.to_string(index=False))
    print("\nCalibration\n", calibration.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tabpfn", "chronos", "evaluate", "all"])
    args = parser.parse_args()
    if args.stage in {"tabpfn", "all"}:
        run_tabpfn()
    if args.stage in {"chronos", "all"}:
        run_chronos()
    if args.stage in {"evaluate", "all"}:
        evaluate()


if __name__ == "__main__":
    main()
