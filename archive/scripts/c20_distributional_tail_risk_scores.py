from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c6_multinode_stageb_risk_model import load_multinode_raw
from c7_pressure_conditioned_deep_ranker import feature_columns
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import build_panel as build_eval_panel
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c12_condition_interaction_diagnostic import monthly_rank
from c16_da_congestion_shape_diagnostic import add_da_congestion_shape, monthly_alert_mask


ROOT = Path(__file__).resolve().parents[1]
EVAL_PRED = ROOT / "c8_vmd_enhanced_sequence_diagnostic" / "c8_vmd_sequence_predictions.csv"
OUT = ROOT / "c20_distributional_tail_risk_scores"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
QUANTILES = [0.05, 0.10, 0.20]
SEED = 20260521


def load_target_series() -> pd.DataFrame:
    paths = sorted(glob.glob("D:/White_Horse_Lake/*/data/input_price_predict_*.csv"))
    parts = []
    for path in paths:
        part = pd.read_csv(path, usecols=["time", "code", "price_day_ahead", "price_real"])
        part = part[part["code"].eq("YQFDC-01")].copy()
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").drop_duplicates("time", keep="last")
    raw["spread_rt_minus_da"] = raw["price_real"] - raw["price_day_ahead"]
    return raw[["time", "spread_rt_minus_da"]]


def build_long_feature_panel() -> pd.DataFrame:
    raw = load_multinode_raw()
    target = load_target_series()
    df = raw.merge(target, on="time", how="left").sort_values("time").reset_index(drop=True)
    df = df[df["spread_rt_minus_da"].notna()].copy()
    df["month"] = df["time"].dt.to_period("M").astype(str)
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_midday"] = ((hour >= 11) & (hour <= 15.5)).astype(float)
    df["is_morning_ramp"] = ((hour >= 6) & (hour <= 10.5)).astype(float)
    df["is_evening_ramp"] = ((hour >= 17) & (hour <= 21.5)).astype(float)

    s = df["spread_rt_minus_da"]
    for lag in [48, 96, 336, 672]:
        df[f"spread_lag_{lag}"] = s.shift(lag)
    df["spread_roll_7d_mean"] = s.shift(48).rolling(336, min_periods=48).mean()
    df["spread_roll_7d_std"] = s.shift(48).rolling(336, min_periods=48).std()
    df["spread_roll_7d_min"] = s.shift(48).rolling(336, min_periods=48).min()
    df["spread_roll_14d_q10"] = s.shift(48).rolling(672, min_periods=96).quantile(0.10)
    df["hist_tail65_rate_7d"] = (s.shift(48) < -65).rolling(336, min_periods=48).mean()
    df["hist_tail100_rate_7d"] = (s.shift(48) < -100).rolling(336, min_periods=48).mean()
    df["neg_spread_lag_48"] = -df["spread_lag_48"]
    df["neg_spread_lag_96"] = -df["spread_lag_96"]
    df["renewable_pred"] = df[["photo_gene_total_pred", "wind_gene_total_pred", "water_gene_total_pred"]].sum(axis=1)
    df["renewable_share"] = df["renewable_pred"] / df["elec_gene_total_pred"].replace(0, np.nan)
    return df


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    x_test = test[features].replace([np.inf, -np.inf], np.nan)
    usable = x_train.notna().any(axis=0)
    used = list(x_train.columns[usable])
    med = x_train[used].median(numeric_only=True).fillna(0.0)
    return x_train[used].fillna(med).fillna(0.0), x_test[used].fillna(med).fillna(0.0), med


def empirical_cdf(samples: np.ndarray, points: np.ndarray) -> np.ndarray:
    samples = np.sort(samples[np.isfinite(samples)])
    if len(samples) == 0:
        return np.full(len(points), 0.5)
    return np.searchsorted(samples, points, side="right") / len(samples)


def empirical_es(samples: np.ndarray, points: np.ndarray) -> np.ndarray:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.zeros(len(points))
    return np.array([np.maximum(p - samples, 0.0).mean() for p in points])


def fit_fold(long_df: pd.DataFrame, eval_month: str, eval_frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    test_times = set(eval_frame.loc[eval_frame["test_month"].eq(eval_month), "time"])
    test = long_df[long_df["time"].isin(test_times)].copy().sort_values("time").reset_index(drop=True)
    tau = float(eval_frame.loc[eval_frame["test_month"].eq(eval_month), "tau_negative"].iloc[0])
    test_start = pd.Timestamp(f"{eval_month}-01")
    cal_month = (test_start - pd.offsets.MonthBegin(1)).to_period("M").strftime("%Y-%m")
    train_all = long_df[long_df["time"] < test_start].copy()
    cal = train_all[train_all["month"].eq(cal_month)].copy()
    core = train_all[~train_all["month"].eq(cal_month)].copy()
    if len(core) < 3000 or len(cal) < 300:
        core = train_all.iloc[: max(1000, int(len(train_all) * 0.8))].copy()
        cal = train_all.iloc[max(1000, int(len(train_all) * 0.8)) :].copy()
    if cal.empty:
        cal = core.tail(min(1000, len(core))).copy()
        core = core.iloc[: -len(cal)].copy()

    x_core, x_test, _ = prepare_xy(core, test, features)
    x_cal = cal[x_core.columns].replace([np.inf, -np.inf], np.nan)
    med = x_core.median(numeric_only=True).fillna(0.0)
    x_cal = x_cal.fillna(med).fillna(0.0)
    y_core = core["spread_rt_minus_da"].astype(float).to_numpy()
    y_cal = cal["spread_rt_minus_da"].astype(float).to_numpy()

    out = test[["time"]].copy()
    out["test_month"] = eval_month

    point = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=260,
        max_leaf_nodes=31,
        l2_regularization=0.10,
        random_state=SEED,
    )
    point.fit(x_core, y_core)
    mu_cal = point.predict(x_cal)
    mu_test = point.predict(x_test)
    residual = y_cal - mu_cal
    out["c20_hgb_mean"] = mu_test
    out["c20_iso_tail_prob"] = empirical_cdf(residual, tau - mu_test)
    out["c20_iso_expected_shortfall"] = empirical_es(residual, tau - mu_test)

    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            learning_rate=0.045,
            max_iter=260,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            random_state=SEED + int(q * 1000),
        )
        model.fit(x_core, y_core)
        pred = model.predict(x_test)
        tag = f"{int(q * 100):02d}"
        out[f"c20_hgb_q{tag}"] = pred
        out[f"c20_hgb_q{tag}_risk"] = tau - pred

    rf = RandomForestRegressor(
        n_estimators=260,
        max_depth=14,
        min_samples_leaf=5,
        max_features=0.70,
        n_jobs=-1,
        random_state=SEED + 77,
    )
    rf.fit(x_core, y_core)
    tree_preds = np.vstack([tree.predict(x_test) for tree in rf.estimators_])
    out["c20_rf_tail_prob"] = (tree_preds < tau).mean(axis=0)
    out["c20_rf_expected_shortfall"] = np.maximum(tau - tree_preds, 0.0).mean(axis=0)
    for q in QUANTILES:
        tag = f"{int(q * 100):02d}"
        q_pred = np.quantile(tree_preds, q, axis=0)
        out[f"c20_rf_q{tag}"] = q_pred
        out[f"c20_rf_q{tag}_risk"] = tau - q_pred

    y_core_tail = (y_core < tau).astype(int)
    if len(np.unique(y_core_tail)) == 2:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=220,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            random_state=SEED + 101,
        )
        clf.fit(x_core, y_core_tail)
        p_cal = clf.predict_proba(x_cal)[:, 1]
        p_test = clf.predict_proba(x_test)[:, 1]
        y_cal_tail = (y_cal < tau).astype(int)
        if len(np.unique(y_cal_tail)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, y_cal_tail)
            p_test = iso.predict(p_test)
        out["c20_hgb_tail_prob_cls"] = p_test
    else:
        out["c20_hgb_tail_prob_cls"] = out["c20_iso_tail_prob"]

    return out


def add_core_c16_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = add_da_congestion_shape(df).copy()
    df["c16_C12_severity_best__da_cong_memory_interaction_am0p15"] = (
        monthly_rank(df, "C12_severity_best").fillna(0.5) - 0.15 * df["da_cong_memory_interaction"].fillna(0.5)
    )
    df["c16_C12_severity_best__da_neg_memory_interaction_am0p15"] = (
        monthly_rank(df, "C12_severity_best").fillna(0.5) - 0.15 * df["da_neg_memory_interaction"].fillna(0.5)
    )
    df["c16_C13_best_exact__da_neg_memory_interaction_a0p25"] = (
        monthly_rank(df, "C13_best_exact").fillna(0.5) + 0.25 * df["da_neg_memory_interaction"].fillna(0.5)
    )
    df["c16_C13_best_exact__r_da_roll_neg_am0p15"] = (
        monthly_rank(df, "C13_best_exact").fillna(0.5) - 0.15 * df["r_da_roll_neg"].fillna(0.5)
    )
    df["c16_C13_best_exact__r_da_delta_abs_a0p15"] = (
        monthly_rank(df, "C13_best_exact").fillna(0.5) + 0.15 * df["r_da_delta_abs"].fillna(0.5)
    )
    df["C19_best_pair_proxy"] = (
        0.7 * monthly_rank(df, "c16_C12_severity_best__da_cong_memory_interaction_am0p15").fillna(0.5)
        + 0.3 * monthly_rank(df, "c16_C12_severity_best__da_neg_memory_interaction_am0p15").fillna(0.5)
    )
    df["C19_high_recall_pair_proxy"] = (
        0.5 * monthly_rank(df, "c16_C13_best_exact__da_neg_memory_interaction_a0p25").fillna(0.5)
        + 0.5 * monthly_rank(df, "c16_C13_best_exact__r_da_roll_neg_am0p15").fillna(0.5)
    )
    return df


def summarize(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    for scope, sdf0 in scopes.items():
        sdf = sdf0.reset_index(drop=True)
        if sdf.empty:
            continue
        for score in score_cols:
            for budget in BUDGETS:
                alert = monthly_alert_mask(sdf, score, budget)
                rows.append({"scope": scope, "method": score, "budget": budget, **metric_from_mask(sdf, alert)})
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


def per_month_breakdown(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for month, mdf0 in df.groupby("test_month", sort=True):
        mdf = mdf0.reset_index(drop=True)
        for method in methods:
            alert = monthly_alert_mask(mdf, method, 0.30)
            rows.append({"test_month": month, "method": method, **metric_from_mask(mdf, alert)})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    eval_frame = pd.read_csv(EVAL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    eval_core = build_eval_panel().sort_values("time").reset_index(drop=True)
    eval_core = add_core_c16_scores(eval_core)

    long_df = build_long_feature_panel()
    features = feature_columns(long_df)
    months = sorted(eval_core["test_month"].unique())
    fold_preds = [fit_fold(long_df, month, eval_core, features) for month in months]
    dist_pred = pd.concat(fold_preds, ignore_index=True)
    df = eval_core.merge(dist_pred.drop(columns=["test_month"]), on="time", how="left")

    dist_scores = [
        c
        for c in df.columns
        if c.startswith("c20_") and (c.endswith("_risk") or "tail_prob" in c or "expected_shortfall" in c)
    ]
    base_scores = [
        "AllSignals_equal_rank_avg",
        "HGB_q10_distance_two_stage",
        "QIA_two_stage",
        "C12_severity_best",
        "C13_best_exact",
        "C19_best_pair_proxy",
        "C19_high_recall_pair_proxy",
        "c16_C13_best_exact__r_da_delta_abs_a0p15",
    ]
    new_blends = []
    for dist in dist_scores:
        d_rank = monthly_rank(df, dist).fillna(0.5)
        for base in ["C12_severity_best", "C13_best_exact", "C19_best_pair_proxy", "C19_high_recall_pair_proxy"]:
            b_rank = monthly_rank(df, base).fillna(0.5)
            for w in [0.20, 0.35, 0.50, 0.65, 0.80]:
                tag = str(w).replace(".", "p")
                name = f"c20_blend_{dist}__{base}_w{tag}"
                df[name] = w * d_rank + (1 - w) * b_rank
                new_blends.append(name)

    score_cols = [c for c in base_scores + dist_scores + new_blends if c in df.columns]
    metrics = summarize(df, score_cols)
    pressure30 = metrics[
        metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)
    ].sort_values(["pressure_hard_gate", "gate_score"], ascending=[False, False])
    all30 = metrics[
        metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)
    ].sort_values(["all_hard_gate", "gate_score"], ascending=[False, False])
    key_methods = list(dict.fromkeys(base_scores + pressure30["method"].head(10).tolist() + all30["method"].head(5).tolist()))
    month_breakdown = per_month_breakdown(df, key_methods)

    df.to_csv(OUT / "c20_distributional_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c20_metrics.csv", index=False, encoding="utf-8-sig")
    pressure30.to_csv(OUT / "c20_pressure_budget30_ranked.csv", index=False, encoding="utf-8-sig")
    all30.to_csv(OUT / "c20_all_budget30_ranked.csv", index=False, encoding="utf-8-sig")
    month_breakdown.to_csv(OUT / "c20_month_breakdown_budget30.csv", index=False, encoding="utf-8-sig")

    report = [
        "# C20 Distributional Tail-Risk Scores",
        "",
        "C20 tests whether validation-safe distributional signals can add information beyond the C12/C16/C19 score pool. It trains on pre-test history only and evaluates on the locked 2025-09 to 2026-02 window.",
        "",
        "Outputs are lower-tail quantile risk, tail probability, and expected-shortfall-like scores. They are used as risk scores for monthly fixed-budget ranking, not as ordinary point-price forecasts.",
        "",
        "## Pressure 30% Best Rows",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "pressure_hard_gate"], 25),
        "",
        "## All-Month 30% Best Rows",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score", "all_hard_gate"], 20),
        "",
        "## Key Per-Month Breakdown",
        "",
        md_table(month_breakdown.sort_values(["method", "test_month"]), ["test_month", "method", "recall", "excess_share", "false_alert_burden", "gate_score"], 80),
    ]
    (OUT / "c20_distributional_tail_risk_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
