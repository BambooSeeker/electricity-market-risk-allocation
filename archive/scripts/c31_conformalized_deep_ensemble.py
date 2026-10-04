from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c12_condition_interaction_diagnostic import monthly_rank
from c16_da_congestion_shape_diagnostic import monthly_alert_mask
from c20_distributional_tail_risk_scores import EVAL_PRED, build_long_feature_panel
from c22_validation_safe_boundary_enhancement import BONUS_FEATURE, PRED as C20_FULL_PRED, build_base_score
from c27_protected_core_boundary_correction import protected_alert_mask


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c31_conformalized_deep_ensemble"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260522
EPOCHS = 42
BATCH_SIZE = 512
ENSEMBLE_SEEDS = [0, 7, 19, 31]
QUANTILES = np.array([0.05, 0.10, 0.20, 0.50, 0.90, 0.95], dtype=np.float32)
BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    cong_cols = [c for c in df.columns if c.startswith("price_day_ahead_cong_")]
    if cong_cols:
        mat = df[cong_cols].astype(float)
        df["da_cong_negative_pressure"] = (-mat.clip(upper=0)).mean(axis=1)
        df["da_cong_abs_mean"] = mat.abs().mean(axis=1)
        df["da_cong_std"] = mat.std(axis=1)
        df["da_cong_range"] = mat.max(axis=1) - mat.min(axis=1)
        for col in ["da_cong_negative_pressure", "da_cong_abs_mean", "da_cong_std", "da_cong_range"]:
            df[f"{col}_delta48"] = df[col] - df[col].shift(48)
            df[f"{col}_roll7"] = df[col].shift(48).rolling(336, min_periods=48).mean()
    df["risk_proxy_q10"] = -df["spread_roll_14d_q10"]
    df["risk_proxy_tail65"] = df["hist_tail65_rate_7d"]
    df["risk_proxy_tail100"] = df["hist_tail100_rate_7d"]
    df["hist_neg_excess65_7d"] = np.maximum(-65 - df["spread_rt_minus_da"].shift(48), 0).rolling(336, min_periods=48).mean()
    df["hist_neg_excess100_7d"] = np.maximum(-100 - df["spread_rt_minus_da"].shift(48), 0).rolling(336, min_periods=48).mean()
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "spread_lag_48",
        "spread_lag_96",
        "spread_lag_336",
        "spread_lag_672",
        "spread_roll_7d_mean",
        "spread_roll_7d_std",
        "spread_roll_7d_min",
        "spread_roll_14d_q10",
        "hist_tail65_rate_7d",
        "hist_tail100_rate_7d",
        "hist_neg_excess65_7d",
        "hist_neg_excess100_7d",
        "neg_spread_lag_48",
        "neg_spread_lag_96",
        "risk_proxy_q10",
        "risk_proxy_tail65",
        "risk_proxy_tail100",
        "load_day_ahead_pred",
        "net_load_pred",
        "renewable_share",
        "machine_state_sum",
        "cong_abs_max",
        "cong_range",
        "node_price_range",
        "da_cong_negative_pressure",
        "da_cong_abs_mean",
        "da_cong_std",
        "da_cong_range",
        "da_cong_negative_pressure_delta48",
        "da_cong_abs_mean_delta48",
        "da_cong_std_delta48",
        "da_cong_range_delta48",
        "da_cong_negative_pressure_roll7",
        "da_cong_abs_mean_roll7",
        "hour",
        "hour_sin",
        "hour_cos",
        "is_midday",
        "is_morning_ramp",
        "is_evening_ramp",
    ]
    return [c for c in candidates if c in df.columns]


class QuantileMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256, dropout: float = 0.12):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.quant = nn.Linear(hidden // 2, len(QUANTILES))
        self.cls = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        q = torch.sort(self.quant(z), dim=1).values
        logit = self.cls(z).squeeze(-1)
        return q, logit


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor(QUANTILES, device=q_pred.device).view(1, -1)
    err = y.view(-1, 1) - q_pred
    return torch.maximum(qs * err, (qs - 1.0) * err).mean()


def train_one(x_train: np.ndarray, y_train: np.ndarray, tau: float, seed: int) -> QuantileMLP:
    seed_all(seed)
    model = QuantileMLP(x_train.shape[1]).to(DEVICE)
    y_tail = (y_train < tau).astype(np.float32)
    pos = y_tail.sum()
    pos_weight = torch.tensor([(len(y_tail) - pos) / max(pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_tail, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            q, logit = model(xb)
            loss = pinball_loss(q, yb) + 0.30 * bce(logit, tb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict_ensemble(models: list[QuantileMLP], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_parts, p_parts = [], []
    for model in models:
        model.eval()
        qs, ps = [], []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.tensor(x[i : i + 4096], dtype=torch.float32, device=DEVICE)
                q, logit = model(xb)
                qs.append(q.cpu().numpy())
                ps.append(torch.sigmoid(logit).cpu().numpy())
        q_parts.append(np.vstack(qs))
        p_parts.append(np.concatenate(ps))
    return np.mean(q_parts, axis=0), np.mean(p_parts, axis=0)


def conformalize_quantiles(q_test: np.ndarray, q_cal: np.ndarray, y_cal: np.ndarray) -> np.ndarray:
    q_out = q_test.copy()
    if len(y_cal) < 50:
        return q_out
    for j, alpha in enumerate(QUANTILES):
        if alpha <= 0.50:
            residual = q_cal[:, j] - y_cal
            qhat = np.quantile(residual, 1.0 - float(alpha), method="higher")
            q_out[:, j] = q_test[:, j] - qhat
        else:
            residual = y_cal - q_cal[:, j]
            qhat = np.quantile(residual, float(alpha), method="higher")
            q_out[:, j] = q_test[:, j] + qhat
    q_out = np.maximum.accumulate(q_out, axis=1)
    return q_out


def fit_fold(long_df: pd.DataFrame, eval_df: pd.DataFrame, month: str, features: list[str]) -> pd.DataFrame:
    test_times = set(eval_df.loc[eval_df["test_month"].eq(month), "time"])
    test = long_df[long_df["time"].isin(test_times)].copy().reset_index(drop=True)
    test_start = pd.Timestamp(f"{month}-01")
    train = long_df[long_df["time"] < test_start].copy().reset_index(drop=True)
    cal_month = (test_start - pd.offsets.MonthBegin(1)).to_period("M").strftime("%Y-%m")
    cal = train[train["month"].eq(cal_month)].copy().reset_index(drop=True)
    core = train[~train["month"].eq(cal_month)].copy().reset_index(drop=True)
    if len(cal) < 300:
        split = int(len(train) * 0.85)
        core, cal = train.iloc[:split].copy(), train.iloc[split:].copy()
    tau = float(eval_df.loc[eval_df["test_month"].eq(month), "tau_negative"].iloc[0])

    med = core[features].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    scaler = StandardScaler()
    core_x = core[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    scaler.fit(core_x)

    def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = scaler.transform(frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)).astype(np.float32)
        y = frame["spread_rt_minus_da"].to_numpy(dtype=np.float32)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        return x[valid], y[valid]

    x_core, y_core = xy(core)
    x_cal, y_cal = xy(cal)
    x_test, _ = xy(test)
    out = test[["time"]].copy()
    out["test_month"] = month
    for name in ["raw", "conf"]:
        for q in ["q05", "q10", "q20", "q50", "q90", "q95"]:
            out[f"c31_{name}_{q}"] = np.nan
    out["c31_tail_prob_raw"] = np.nan
    out["c31_tail_prob_iso"] = np.nan
    if len(x_core) < 1000 or len(x_test) == 0:
        return out
    models = [train_one(x_core, y_core, tau, SEED + s) for s in ENSEMBLE_SEEDS]
    q_test, p_test = predict_ensemble(models, x_test)
    q_cal, p_cal = predict_ensemble(models, x_cal) if len(x_cal) else (np.empty((0, len(QUANTILES))), np.empty(0))
    q_conf = conformalize_quantiles(q_test, q_cal, y_cal)
    p_iso = p_test.copy()
    if len(x_cal) > 50:
        y_tail = (y_cal < tau).astype(int)
        if len(np.unique(y_tail)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, y_tail)
            p_iso = iso.predict(p_test)

    q_names = ["q05", "q10", "q20", "q50", "q90", "q95"]
    for j, name in enumerate(q_names):
        out[f"c31_raw_{name}"] = q_test[:, j]
        out[f"c31_conf_{name}"] = q_conf[:, j]
    out["c31_tail_prob_raw"] = p_test
    out["c31_tail_prob_iso"] = p_iso
    return out


def fixed_budget_metrics(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    for scope, sdf0 in scopes.items():
        for col in score_cols:
            for budget in BUDGETS:
                tmp = sdf0[["test_month", "negative_tail", "negative_excess", col]].dropna().copy().reset_index(drop=True)
                tmp = tmp.rename(columns={col: "_score"})
                alert = monthly_alert_mask(tmp, "_score", budget)
                met = metric_from_mask(tmp, alert)
                rows.append({"scope": scope, "method": col, "budget": budget, **met})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def probabilistic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    y_col = "spread_rt_minus_da"
    for scope, sdf in scopes.items():
        for prefix in ["raw", "conf"]:
            q_cols = [f"c31_{prefix}_{q}" for q in ["q05", "q10", "q20", "q50", "q90", "q95"]]
            tmp = sdf[[y_col] + q_cols].dropna().copy()
            if tmp.empty:
                continue
            y = tmp[y_col].to_numpy()
            q = tmp[q_cols].to_numpy()
            pinballs = []
            for j, alpha in enumerate(QUANTILES):
                err = y - q[:, j]
                pinballs.append(np.maximum(alpha * err, (alpha - 1.0) * err).mean())
            cover90 = ((y >= q[:, 0]) & (y <= q[:, -1])).mean()
            width90 = (q[:, -1] - q[:, 0]).mean()
            cover80 = ((y >= q[:, 1]) & (y <= q[:, 4])).mean()
            width80 = (q[:, 4] - q[:, 1]).mean()
            rows.append(
                {
                    "scope": scope,
                    "method": f"c31_{prefix}_quantiles",
                    "mean_pinball": float(np.mean(pinballs)),
                    "pinball_q05": float(pinballs[0]),
                    "pinball_q10": float(pinballs[1]),
                    "pinball_q20": float(pinballs[2]),
                    "coverage_90": float(cover90),
                    "width_90": float(width90),
                    "coverage_80": float(cover80),
                    "width_80": float(width80),
                }
            )
        for prob_col in ["c31_tail_prob_raw", "c31_tail_prob_iso"]:
            tmp = sdf[["negative_tail", prob_col]].dropna().copy()
            if tmp.empty:
                continue
            p = np.clip(tmp[prob_col].to_numpy(), 1e-6, 1 - 1e-6)
            y = tmp["negative_tail"].to_numpy()
            brier = np.mean((p - y) ** 2)
            logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            rows.append({"scope": scope, "method": prob_col, "brier": float(brier), "logloss": float(logloss)})
    return pd.DataFrame(rows)


def add_blends(full: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    full["base_score"] = build_base_score(full)
    full["boundary_bonus"] = monthly_rank(full, BONUS_FEATURE).fillna(0.5)
    full["protected_core_alert_score_proxy"] = full["base_score"] + 0.20 * full["boundary_bonus"]
    tau = full["tau_negative"].to_numpy()
    for prefix in ["raw", "conf"]:
        full[f"c31_{prefix}_q05_risk"] = tau - full[f"c31_{prefix}_q05"].to_numpy()
        full[f"c31_{prefix}_q10_risk"] = tau - full[f"c31_{prefix}_q10"].to_numpy()
        full[f"c31_{prefix}_q20_risk"] = tau - full[f"c31_{prefix}_q20"].to_numpy()
        full[f"c31_{prefix}_expected_shortfall"] = np.maximum(tau.reshape(-1, 1) - full[[f"c31_{prefix}_q05", f"c31_{prefix}_q10", f"c31_{prefix}_q20"]].to_numpy(), 0).mean(axis=1)
    for col in [
        "c31_tail_prob_iso",
        "c31_conf_q20_risk",
        "c31_conf_expected_shortfall",
        "c31_raw_q20_risk",
    ]:
        r = full[col].rank(method="average", pct=True)
        full[f"c31_blend_{col}_base_w0p2"] = 0.2 * r + 0.8 * full["base_score"]
        full[f"c31_blend_{col}_base_w0p35"] = 0.35 * r + 0.65 * full["base_score"]
    return full


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    seed_all(SEED)
    eval_df = pd.read_csv(EVAL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    long_df = add_features(build_long_feature_panel())
    features = feature_cols(long_df)
    parts = [fit_fold(long_df, eval_df, month, features) for month in sorted(eval_df["test_month"].unique())]
    pred = pd.concat(parts, ignore_index=True)
    c20_full = pd.read_csv(C20_FULL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    full = c20_full.merge(pred.drop(columns=["test_month"]), on="time", how="left")
    full = add_blends(full)

    score_cols = [
        "c31_tail_prob_raw",
        "c31_tail_prob_iso",
        "c31_raw_q05_risk",
        "c31_raw_q10_risk",
        "c31_raw_q20_risk",
        "c31_conf_q05_risk",
        "c31_conf_q10_risk",
        "c31_conf_q20_risk",
        "c31_conf_expected_shortfall",
    ] + [c for c in full.columns if c.startswith("c31_blend_")]
    risk_metrics = fixed_budget_metrics(full, score_cols)
    prob_metrics = probabilistic_metrics(full)

    # Compare protected-core with conformal deep scores under the same fixed-budget metric.
    tmp = full.copy()
    tmp["base_rank_pct"] = tmp.groupby("test_month", group_keys=False)["base_score"].rank(method="first", ascending=False, pct=True)
    tmp["protected_core_score_for_table"] = tmp["base_score"] + 0.20 * tmp["boundary_bonus"]
    pc_alert = protected_alert_mask(tmp, "base_score", "protected_core_score_for_table", 0.30, 0.225, 0.40)
    pc_rows = []
    for scope, sdf in {
        "all_available_months": tmp,
        "pressure_available_months": tmp[tmp["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": tmp[~tmp["test_month"].isin(PRESSURE_MONTHS)],
    }.items():
        met = metric_from_mask(
            sdf[["test_month", "negative_tail", "negative_excess"]].reset_index(drop=True),
            pc_alert.loc[sdf.index].reset_index(drop=True),
        )
        met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
        pc_rows.append({"scope": scope, "method": "protected_core_reference", "budget": 0.30, **met})
    pc_metrics = pd.DataFrame(pc_rows)
    risk_metrics = pd.concat([risk_metrics, pc_metrics], ignore_index=True)

    full.to_csv(OUT / "c31_conformalized_deep_ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    risk_metrics.to_csv(OUT / "c31_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")
    prob_metrics.to_csv(OUT / "c31_probabilistic_metrics.csv", index=False, encoding="utf-8-sig")

    pressure30 = risk_metrics[risk_metrics["scope"].eq("pressure_available_months") & risk_metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    all30 = risk_metrics[risk_metrics["scope"].eq("all_available_months") & risk_metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    report = [
        "# C31 Conformalized Deep Ensemble",
        "",
        "Purpose: add a modern probabilistic deep ensemble baseline with conformalized quantiles and probability calibration. This is a technical-hardness comparison, not a new-model claim.",
        "",
        f"Device: `{DEVICE}`",
        "",
        f"Features: {len(features)} risk/mechanism state features.",
        "",
        "## Pressure 30% Fixed-Budget Results",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## All-Month 30% Fixed-Budget Results",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## Probabilistic Scores",
        "",
        md_table(prob_metrics.fillna(""), ["scope", "method", "mean_pinball", "coverage_90", "width_90", "coverage_80", "width_80", "brier", "logloss"], 30),
    ]
    (OUT / "c31_conformalized_deep_ensemble_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
