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
from c16_da_congestion_shape_diagnostic import monthly_alert_mask
from c20_distributional_tail_risk_scores import EVAL_PRED, build_long_feature_panel
from c22_validation_safe_boundary_enhancement import PRED as C20_FULL_PRED, build_base_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c32_ssm_like_risk_sequence_baseline"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260522
SEQ_LEN = 48
EPOCHS = 18
BATCH_SIZE = 256
ENSEMBLE_SEEDS = [0]
QUANTILES = np.array([0.05, 0.10, 0.20], dtype=np.float32)
BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["risk_proxy_q20"] = -df["spread_roll_14d_q10"]
    df["risk_proxy_tail65"] = df["hist_tail65_rate_7d"]
    df["risk_proxy_tail100"] = df["hist_tail100_rate_7d"]
    df["hist_neg_excess65_7d"] = np.maximum(-65 - df["spread_rt_minus_da"].shift(48), 0).rolling(336, min_periods=48).mean()
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
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
        "neg_spread_lag_48",
        "neg_spread_lag_96",
        "risk_proxy_q20",
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
        "hour_sin",
        "hour_cos",
        "is_midday",
        "is_morning_ramp",
        "is_evening_ramp",
    ]
    return [c for c in cols if c in df.columns]


def make_sequences(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[features].to_numpy(dtype=np.float32)
    y = df["spread_rt_minus_da"].to_numpy(dtype=np.float32)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    xs, ys, pos = [], [], []
    for i in range(SEQ_LEN - 1, len(df)):
        sl = slice(i - SEQ_LEN + 1, i + 1)
        if valid[sl].all():
            xs.append(x[sl])
            ys.append(y[i])
            pos.append(i)
    if not xs:
        return np.empty((0, SEQ_LEN, len(features)), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), np.asarray(pos, dtype=int)


class GatedSSMLayer(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden)
        self.gate_proj = nn.Linear(n_features, hidden)
        self.decay_logit = nn.Parameter(torch.zeros(hidden))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        h = x.new_zeros(batch, decay.shape[1])
        outs = []
        cand_all = torch.tanh(self.in_proj(x))
        gate_all = torch.sigmoid(self.gate_proj(x))
        for t in range(steps):
            candidate = gate_all[:, t, :] * cand_all[:, t, :]
            h = decay * h + (1.0 - decay) * candidate
            outs.append(h)
        return self.out_norm(torch.stack(outs, dim=1))


class SSMRiskNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.12):
        super().__init__()
        self.ssm1 = GatedSSMLayer(n_features, hidden)
        self.ssm2 = GatedSSMLayer(hidden, hidden)
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.quant = nn.Linear(hidden, len(QUANTILES))
        self.cls = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ssm1(x)
        h = self.ssm2(self.mix(h))
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        z = torch.sum(h * w.unsqueeze(-1), dim=1)
        z = self.head(z)
        q = torch.sort(self.quant(z), dim=1).values
        logit = self.cls(z).squeeze(-1)
        return q, logit


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor(QUANTILES, device=q_pred.device).view(1, -1)
    err = y.view(-1, 1) - q_pred
    return torch.maximum(qs * err, (qs - 1.0) * err).mean()


def train_one(x_train: np.ndarray, y_train: np.ndarray, tau: float, seed: int) -> SSMRiskNet:
    seed_all(seed)
    model = SSMRiskNet(x_train.shape[-1]).to(DEVICE)
    y_tail = (y_train < tau).astype(np.float32)
    pos = y_tail.sum()
    pos_weight = torch.tensor([(len(y_tail) - pos) / max(pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train), torch.tensor(y_tail))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            q, logit = model(xb)
            loss = pinball_loss(q, yb) + 0.35 * bce(logit, tb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict(models: list[SSMRiskNet], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_parts, p_parts = [], []
    for model in models:
        model.eval()
        qs, ps = [], []
        with torch.no_grad():
            for i in range(0, len(x), 1024):
                xb = torch.tensor(x[i : i + 1024], dtype=torch.float32, device=DEVICE)
                q, logit = model(xb)
                qs.append(q.cpu().numpy())
                ps.append(torch.sigmoid(logit).cpu().numpy())
        q_parts.append(np.vstack(qs))
        p_parts.append(np.concatenate(ps))
    return np.mean(q_parts, axis=0), np.mean(p_parts, axis=0)


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
    scaler.fit(core[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0))

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[features] = scaler.transform(frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0))
        return out

    core_p, cal_p, test_p = prep(core), prep(cal), prep(test)
    x_train, y_train, _ = make_sequences(core_p, features)
    x_cal, y_cal, _ = make_sequences(cal_p, features)
    x_test, _, pos_test = make_sequences(test_p, features)
    out = test[["time"]].copy()
    out["test_month"] = month
    for col in ["c32_ssm_tail_prob", "c32_ssm_tail_prob_iso", "c32_ssm_q05_risk", "c32_ssm_q10_risk", "c32_ssm_q20_risk", "c32_ssm_expected_shortfall"]:
        out[col] = np.nan
    if len(x_train) < 1000 or len(x_test) == 0:
        return out
    models = [train_one(x_train, y_train, tau, SEED + s) for s in ENSEMBLE_SEEDS]
    q_test, p_test = predict(models, x_test)
    p_iso = p_test.copy()
    if len(x_cal) > 50:
        _, p_cal = predict(models, x_cal)
        y_tail = (y_cal < tau).astype(int)
        if len(np.unique(y_tail)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, y_tail)
            p_iso = iso.predict(p_test)
    out.loc[pos_test, "c32_ssm_tail_prob"] = p_test
    out.loc[pos_test, "c32_ssm_tail_prob_iso"] = p_iso
    out.loc[pos_test, "c32_ssm_q05_risk"] = tau - q_test[:, 0]
    out.loc[pos_test, "c32_ssm_q10_risk"] = tau - q_test[:, 1]
    out.loc[pos_test, "c32_ssm_q20_risk"] = tau - q_test[:, 2]
    out.loc[pos_test, "c32_ssm_expected_shortfall"] = np.maximum(tau - q_test, 0).mean(axis=1)
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
                met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
                rows.append({"scope": scope, "method": col, "budget": budget, **met})
    return pd.DataFrame(rows)


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
    long_df = add_state_features(build_long_feature_panel())
    features = feature_cols(long_df)
    parts = [fit_fold(long_df, eval_df, month, features) for month in sorted(eval_df["test_month"].unique())]
    pred = pd.concat(parts, ignore_index=True)
    c20_full = pd.read_csv(C20_FULL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    full = c20_full.merge(pred.drop(columns=["test_month"]), on="time", how="left")
    base = build_base_score(full)
    for col in ["c32_ssm_tail_prob_iso", "c32_ssm_q20_risk", "c32_ssm_expected_shortfall"]:
        r = full[col].rank(method="average", pct=True)
        full[f"c32_blend_{col}_base_w0p2"] = 0.2 * r + 0.8 * base
        full[f"c32_blend_{col}_base_w0p35"] = 0.35 * r + 0.65 * base
    score_cols = [
        "c32_ssm_tail_prob",
        "c32_ssm_tail_prob_iso",
        "c32_ssm_q05_risk",
        "c32_ssm_q10_risk",
        "c32_ssm_q20_risk",
        "c32_ssm_expected_shortfall",
    ] + [c for c in full.columns if c.startswith("c32_blend_")]
    metrics = fixed_budget_metrics(full, score_cols)
    full.to_csv(OUT / "c32_ssm_like_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c32_ssm_like_metrics.csv", index=False, encoding="utf-8-sig")
    pressure30 = metrics[metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    report = [
        "# C32 SSM-Like Risk-State Sequence Baseline",
        "",
        "Purpose: add a state-space-style sequence baseline without claiming a Mamba implementation, because `mamba_ssm` is not installed. The model uses gated diagonal state recurrences over risk/mechanism-state sequences.",
        "",
        f"Device: `{DEVICE}`",
        "",
        f"Features: {len(features)}; sequence length: {SEQ_LEN}; ensemble seeds: {len(ENSEMBLE_SEEDS)}.",
        "",
        "## Pressure 30% Fixed-Budget Results",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## All-Month 30% Fixed-Budget Results",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
    ]
    (OUT / "c32_ssm_like_risk_sequence_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
