from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from c2_deep_severity_sequence_model import PRESSURE_MONTHS, rank01
from c3_validation_safe_diversified_policy import metric_from_mask


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_family_complementarity_audit" / "merged_family_scores.csv"
OUT = ROOT / "c6_multinode_stageb_risk_model"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
RAW_COLS = [
    "time",
    "code",
    "machine_state",
    "price_day_ahead",
    "price_day_ahead_energy",
    "price_day_ahead_cong",
    "load_day_ahead_pred",
    "elec_exter_plan",
    "elec_fix_out_plan",
    "energy_hydro_renewable",
    "elec_gene_total_pred",
    "photo_gene_total_pred",
    "wind_gene_total_pred",
    "water_gene_total_pred",
    "elec_day_ahead_all",
    "price_day_ahead_load",
]


def load_multinode_raw() -> pd.DataFrame:
    paths = sorted(glob.glob("D:/White_Horse_Lake/*/data/input_price_predict_*.csv"))
    parts = [pd.read_csv(path, usecols=lambda c: c in RAW_COLS) for path in paths]
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values(["time", "code"]).drop_duplicates(["time", "code"], keep="last")

    wide_cols = [
        "machine_state",
        "price_day_ahead",
        "price_day_ahead_energy",
        "price_day_ahead_cong",
        "elec_fix_out_plan",
        "price_day_ahead_load",
    ]
    wide = raw.pivot(index="time", columns="code", values=wide_cols)
    wide.columns = [f"{var}_{code}" for var, code in wide.columns]
    wide = wide.reset_index()

    # System-level forecasts are duplicated by code; take the first available value per time.
    sys_cols = [
        "load_day_ahead_pred",
        "elec_exter_plan",
        "energy_hydro_renewable",
        "elec_gene_total_pred",
        "photo_gene_total_pred",
        "wind_gene_total_pred",
        "water_gene_total_pred",
        "elec_day_ahead_all",
    ]
    sys = raw.groupby("time", as_index=False)[sys_cols].first()
    out = wide.merge(sys, on="time", how="left")

    cong_cols = [c for c in out.columns if c.startswith("price_day_ahead_cong_")]
    price_cols = [c for c in out.columns if c.startswith("price_day_ahead_") and not c.startswith("price_day_ahead_cong") and not c.startswith("price_day_ahead_energy") and not c.startswith("price_day_ahead_load")]
    fix_cols = [c for c in out.columns if c.startswith("elec_fix_out_plan_")]
    state_cols = [c for c in out.columns if c.startswith("machine_state_")]

    out["cong_mean"] = out[cong_cols].mean(axis=1)
    out["cong_std"] = out[cong_cols].std(axis=1)
    out["cong_min"] = out[cong_cols].min(axis=1)
    out["cong_max"] = out[cong_cols].max(axis=1)
    out["cong_range"] = out["cong_max"] - out["cong_min"]
    out["cong_abs_max"] = out[cong_cols].abs().max(axis=1)
    out["node_price_std"] = out[price_cols].std(axis=1)
    out["node_price_range"] = out[price_cols].max(axis=1) - out[price_cols].min(axis=1)
    out["fix_out_sum"] = out[fix_cols].sum(axis=1)
    out["machine_state_sum"] = out[state_cols].sum(axis=1)
    out["net_load_pred"] = out["load_day_ahead_pred"] - out["wind_gene_total_pred"].fillna(0) - out["photo_gene_total_pred"].fillna(0)
    return out


def build_panel() -> pd.DataFrame:
    base = pd.read_csv(BASE, parse_dates=["time"])
    raw = load_multinode_raw()
    df = base.merge(raw, on="time", how="left")
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_midday"] = ((hour >= 11) & (hour <= 15.5)).astype(float)
    df["is_morning_ramp"] = ((hour >= 6) & (hour <= 10.5)).astype(float)
    df["cong_range_x_midday"] = df["cong_range"] * df["is_midday"]
    df["fix_out_x_midday"] = df["fix_out_sum"] * df["is_midday"]
    return df.sort_values("time").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "machine_state_",
        "price_day_ahead_",
        "elec_fix_out_plan_",
    )
    cols = [
        "QIA_two_stage",
        "HGB_q10_distance_two_stage",
        "LightGBM_q10_distance_two_stage",
        "AllSignals_equal_rank_avg",
        "load_day_ahead_pred",
        "elec_exter_plan",
        "energy_hydro_renewable",
        "elec_gene_total_pred",
        "photo_gene_total_pred",
        "wind_gene_total_pred",
        "water_gene_total_pred",
        "elec_day_ahead_all",
        "cong_mean",
        "cong_std",
        "cong_min",
        "cong_max",
        "cong_range",
        "cong_abs_max",
        "node_price_std",
        "node_price_range",
        "fix_out_sum",
        "machine_state_sum",
        "net_load_pred",
        "hour_sin",
        "hour_cos",
        "is_midday",
        "is_morning_ramp",
        "cong_range_x_midday",
        "fix_out_x_midday",
    ]
    cols += [c for c in df.columns if c.startswith(prefixes)]
    return [c for c in cols if c in df.columns]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    med = train[features].median(numeric_only=True)
    x_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(med)
    x_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(med)
    y = train["negative_tail"].astype(int).to_numpy()

    out = test[["fold", "test_month", "time", "negative_tail", "negative_excess"]].copy()
    if len(np.unique(y)) < 2:
        for col in ["c6_hgb", "c6_rf", "c6_extra"]:
            out[col] = test["HGB_q10_distance_two_stage"].to_numpy()
        return out

    models = {
        "c6_hgb": HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=320,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            random_state=20260520,
        ),
        "c6_rf": RandomForestClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=20260521,
            n_jobs=-1,
        ),
        "c6_extra": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=20260522,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        model.fit(x_train, y)
        out[name] = model.predict_proba(x_test)[:, 1]
    return out


def evaluate_monthly_budget(panel: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    scopes = {
        "all_available_months": panel,
        "pressure_available_months": panel[panel["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": panel[~panel["test_month"].isin(PRESSURE_MONTHS)],
    }
    rows = []
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for score in score_cols:
            for budget in BUDGETS:
                masks, parts = [], []
                for _, mdf in sdf.groupby("test_month", sort=True):
                    mdf = mdf.reset_index(drop=True)
                    order = np.argsort(-mdf[score].to_numpy())
                    mask = np.zeros(len(mdf), dtype=bool)
                    mask[order[: max(1, int(np.ceil(len(mdf) * budget)))]] = True
                    masks.append(mask)
                    parts.append(mdf)
                joined = pd.concat(parts, ignore_index=True)
                alert = np.concatenate(masks)
                met = metric_from_mask(joined, alert)
                rows.append({"scope": scope, "method": score, "budget": budget, **met})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def main() -> None:
    df = build_panel()
    features = feature_columns(df)
    months = sorted(df["test_month"].astype(str).unique())
    parts = []
    for month in months:
        train = df[df["test_month"].astype(str) < month].copy()
        test = df[df["test_month"].astype(str).eq(month)].copy()
        if train.empty:
            continue
        parts.append(fit_predict(train, test, features))
    pred = pd.concat(parts, ignore_index=True)
    base_cols = ["fold", "test_month", "time", "QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]
    pred = pred.merge(df[base_cols], on=["fold", "test_month", "time"], how="left")

    pred["c6_hgb_blend"] = 0.55 * rank01(pred["HGB_q10_distance_two_stage"].to_numpy()) + 0.45 * rank01(pred["c6_hgb"].to_numpy())
    pred["c6_rf_blend"] = 0.55 * rank01(pred["HGB_q10_distance_two_stage"].to_numpy()) + 0.45 * rank01(pred["c6_rf"].to_numpy())
    pred["c6_extra_blend"] = 0.55 * rank01(pred["HGB_q10_distance_two_stage"].to_numpy()) + 0.45 * rank01(pred["c6_extra"].to_numpy())
    pred["c6_all_blend"] = (
        0.35 * rank01(pred["HGB_q10_distance_two_stage"].to_numpy())
        + 0.25 * rank01(pred["c6_hgb"].to_numpy())
        + 0.20 * rank01(pred["c6_rf"].to_numpy())
        + 0.20 * rank01(pred["c6_extra"].to_numpy())
    )

    score_cols = [
        "QIA_two_stage",
        "HGB_q10_distance_two_stage",
        "AllSignals_equal_rank_avg",
        "c6_hgb",
        "c6_rf",
        "c6_extra",
        "c6_hgb_blend",
        "c6_rf_blend",
        "c6_extra_blend",
        "c6_all_blend",
    ]
    metrics = evaluate_monthly_budget(pred, score_cols)
    pred.to_csv(OUT / "c6_multinode_stageb_predictions.csv", index=False, encoding="utf-8-sig")
    pd.Series(features, name="feature").to_csv(OUT / "c6_feature_columns.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c6_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")

    key = metrics[metrics["budget"].eq(0.30)].sort_values(["scope", "gate_score"], ascending=[True, False])
    key.to_csv(OUT / "c6_key_budget30_rows.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = [
        "# C6 Multi-Node Stage-B Risk Model",
        "",
        "Uses all available unit/node day-ahead features to approximate network-state information.",
        "",
        f"Feature count: {len(features)}",
        "",
        "## 30% Fixed-Budget Key Rows",
        "",
        md_table(key, ["scope", "method", "budget", "recall", "excess_share", "false_alert_burden", "gate_score"], 40),
    ]
    (OUT / "c6_multinode_stageb_risk_model_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
