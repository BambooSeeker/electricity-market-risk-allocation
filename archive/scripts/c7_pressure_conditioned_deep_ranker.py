from __future__ import annotations

from pathlib import Path
import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from c2_deep_severity_sequence_model import PRESSURE_MONTHS, rank01
from c3_validation_safe_diversified_policy import metric_from_mask
from c6_multinode_stageb_risk_model import load_multinode_raw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_family_complementarity_audit" / "merged_family_scores.csv"
C2 = ROOT / "c2_deep_severity_sequence_model" / "c2_deep_sequence_predictions.csv"
OUT = ROOT / "c7_pressure_conditioned_deep_ranker"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260521
BUDGETS = [0.20, 0.30]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(SEED)


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


def build_panel() -> pd.DataFrame:
    raw_feat = load_multinode_raw()
    target = load_target_series()
    df = raw_feat.merge(target, on="time", how="left").sort_values("time").reset_index(drop=True)
    df = df[df["spread_rt_minus_da"].notna()].copy()
    base = pd.read_csv(BASE, parse_dates=["time"])
    labels = base[["time", "fold", "test_month", "negative_tail", "negative_excess", "QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]]
    df = df.merge(labels, on="time", how="left")
    df["month"] = df["time"].dt.to_period("M").astype(str)
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_midday"] = ((hour >= 11) & (hour <= 15.5)).astype(float)
    df["is_morning_ramp"] = ((hour >= 6) & (hour <= 10.5)).astype(float)

    # Historical, information-safe spread features. Use previous-day and previous-week style lags.
    s = df["spread_rt_minus_da"]
    for lag in [48, 96, 336]:
        df[f"spread_lag_{lag}"] = s.shift(lag)
    df["spread_roll_7d_mean"] = s.shift(48).rolling(336, min_periods=48).mean()
    df["spread_roll_7d_std"] = s.shift(48).rolling(336, min_periods=48).std()
    df["spread_roll_7d_min"] = s.shift(48).rolling(336, min_periods=48).min()
    df["hist_tail65_rate_7d"] = (s.shift(48) < -65).rolling(336, min_periods=48).mean()
    df["hist_tail100_rate_7d"] = (s.shift(48) < -100).rolling(336, min_periods=48).mean()

    # Pseudo labels for months before official SLTS labels are available.
    df["pseudo_tail"] = (df["spread_rt_minus_da"] < -70).astype(float)
    df["pseudo_excess"] = np.clip(-70 - df["spread_rt_minus_da"], 0, None)
    df["train_tail"] = df["negative_tail"].fillna(df["pseudo_tail"]).astype(float)
    df["train_excess"] = df["negative_excess"].fillna(df["pseudo_excess"]).astype(float)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    manual = [
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
        "spread_lag_48",
        "spread_lag_96",
        "spread_lag_336",
        "spread_roll_7d_mean",
        "spread_roll_7d_std",
        "spread_roll_7d_min",
        "hist_tail65_rate_7d",
        "hist_tail100_rate_7d",
    ]
    wide = [
        c
        for c in df.columns
        if c.startswith("price_day_ahead_cong_")
        or c.startswith("elec_fix_out_plan_")
        or c.startswith("machine_state_")
    ]
    return [c for c in manual + wide if c in df.columns]


class RankNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 192, dropout: float = 0.18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.cls = nn.Linear(hidden // 2, 1)
        self.sev = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.GELU(), nn.Linear(hidden // 4, 1), nn.Softplus())

    def forward(self, x):
        h = self.net(x)
        logit = self.cls(h).squeeze(-1)
        sev = self.sev(h).squeeze(-1)
        return logit, sev


def pairwise_loss(score: torch.Tensor, y: torch.Tensor, excess: torch.Tensor) -> torch.Tensor:
    pos = torch.where((y > 0.5) & (excess > 0))[0]
    neg = torch.where(y < 0.5)[0]
    if len(pos) == 0 or len(neg) == 0:
        return score.new_tensor(0.0)
    pos = pos[torch.argsort(excess[pos], descending=True)[: min(96, len(pos))]]
    neg = neg[torch.randperm(len(neg), device=score.device)[: min(160, len(neg))]]
    diff = score[pos].unsqueeze(1) - score[neg].unsqueeze(0)
    w = torch.clamp(torch.log1p(excess[pos]).unsqueeze(1) / 5.0, 0.2, 5.0)
    return (nn.functional.softplus(-diff) * w).mean()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], cfg_name: str, tail_weight: float, rank_weight: float) -> pd.DataFrame:
    x_train_raw = train[features].replace([np.inf, -np.inf], np.nan)
    x_test_raw = test[features].replace([np.inf, -np.inf], np.nan)
    usable = x_train_raw.notna().any(axis=0)
    used_features = list(x_train_raw.columns[usable])
    if not used_features:
        raise ValueError(f"{cfg_name}: no usable features for {test['month'].iloc[0]}")
    med = x_train_raw[used_features].median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw[used_features].fillna(med).fillna(0.0)
    x_test = x_test_raw[used_features].fillna(med).fillna(0.0)
    scaler = StandardScaler()
    x_train = np.nan_to_num(scaler.fit_transform(x_train), nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(scaler.transform(x_test), nan=0.0, posinf=0.0, neginf=0.0)
    y = train["train_tail"].astype(float).to_numpy()
    excess = train["train_excess"].astype(float).to_numpy()

    model = RankNet(x_train.shape[1]).to(DEVICE)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(excess, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    pos_rate = max(y.mean(), 1e-4)
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate, device=DEVICE, dtype=torch.float32).clamp(1, 30)
    for _ in range(80):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(DEVICE), yb.to(DEVICE), eb.to(DEVICE)
            logit, sev = model(xb)
            bce = nn.functional.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight, reduction="none")
            sev_w = 1.0 + tail_weight * torch.clamp(torch.log1p(eb) / 5.0, 0, 5)
            cls_loss = (bce * sev_w).mean()
            pos_mask = yb > 0.5
            sev_loss = nn.functional.smooth_l1_loss(sev[pos_mask], torch.log1p(eb[pos_mask])) if pos_mask.any() else sev.new_tensor(0.0)
            risk_score = logit + 0.35 * sev
            loss = cls_loss + 0.35 * sev_loss + rank_weight * pairwise_loss(risk_score, yb, eb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    probs, sevs = [], []
    with torch.no_grad():
        xt = torch.tensor(x_test, dtype=torch.float32)
        for start in range(0, len(xt), 1024):
            logit, sev = model(xt[start : start + 1024].to(DEVICE))
            probs.append(torch.sigmoid(logit).cpu().numpy())
            sevs.append(sev.cpu().numpy())
    out = test[["time", "month", "negative_tail", "negative_excess"]].copy()
    out[f"{cfg_name}_prob"] = np.concatenate(probs)
    out[f"{cfg_name}_risk"] = np.concatenate(probs) * (1 + np.concatenate(sevs))
    if out[[f"{cfg_name}_prob", f"{cfg_name}_risk"]].isna().any().any():
        raise ValueError(f"{cfg_name}: NaN predictions for {test['month'].iloc[0]}")
    return out


def evaluate_monthly(panel: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    scopes = {
        "all_available_months": panel,
        "pressure_available_months": panel[panel["month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": panel[~panel["month"].isin(PRESSURE_MONTHS)],
    }
    rows = []
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for score in score_cols:
            for budget in BUDGETS:
                masks, parts = [], []
                for _, mdf in sdf.groupby("month", sort=True):
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
    eval_months = sorted(df[df["negative_tail"].notna()]["month"].unique())
    parts = []
    configs = [
        ("c7_tw4_rw30", 4.0, 0.30),
        ("c7_tw7_rw45", 7.0, 0.45),
        ("c7_tw10_rw60", 10.0, 0.60),
    ]
    for month in eval_months:
        train = df[df["month"] < month].copy()
        test = df[df["month"].eq(month)].copy()
        test = test[test["negative_tail"].notna()].copy()
        if train.empty or test.empty:
            continue
        month_out = test[["time", "month", "negative_tail", "negative_excess", "QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]].copy()
        for cfg_name, tail_weight, rank_weight in configs:
            pred = fit_predict(train, test, features, cfg_name, tail_weight, rank_weight)
            month_out = month_out.merge(pred.drop(columns=["negative_tail", "negative_excess", "month"]), on="time", how="left")
        parts.append(month_out)
    pred = pd.concat(parts, ignore_index=True)
    for cfg_name, _, _ in configs:
        pred[f"{cfg_name}_risk_hgb_blend"] = 0.55 * rank01(pred["HGB_q10_distance_two_stage"].to_numpy()) + 0.45 * rank01(pred[f"{cfg_name}_risk"].to_numpy())
        pred[f"{cfg_name}_risk_all_blend"] = 0.45 * rank01(pred["AllSignals_equal_rank_avg"].to_numpy()) + 0.35 * rank01(pred[f"{cfg_name}_risk"].to_numpy()) + 0.20 * rank01(pred[f"{cfg_name}_prob"].to_numpy())

    score_cols = ["QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]
    for cfg_name, _, _ in configs:
        score_cols += [f"{cfg_name}_prob", f"{cfg_name}_risk", f"{cfg_name}_risk_hgb_blend", f"{cfg_name}_risk_all_blend"]
    metrics = evaluate_monthly(pred, score_cols)
    pred.to_csv(OUT / "c7_pressure_conditioned_predictions.csv", index=False, encoding="utf-8-sig")
    pd.Series(features, name="feature").to_csv(OUT / "c7_feature_columns.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c7_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")
    key = metrics[metrics["budget"].eq(0.30)].sort_values(["scope", "gate_score"], ascending=[True, False])
    key.to_csv(OUT / "c7_key_budget30_rows.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 50) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = [
        "# C7 Pressure-Conditioned Deep Ranker",
        "",
        f"Device: {DEVICE}",
        "",
        "Uses longer raw history, multi-node Stage-B features, previous-day/week spread features, severity-weighted BCE, and pairwise ranking loss.",
        "",
        "## 30% Monthly Fixed-Budget Key Rows",
        "",
        md_table(key, ["scope", "method", "budget", "recall", "excess_share", "false_alert_burden", "gate_score"], 60),
    ]
    (OUT / "c7_pressure_conditioned_deep_ranker_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
