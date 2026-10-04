from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c16_da_congestion_shape_diagnostic import monthly_alert_mask
from c20_distributional_tail_risk_scores import EVAL_PRED, build_long_feature_panel
from c22_validation_safe_boundary_enhancement import PRED as C20_FULL_PRED, build_base_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c33_conditional_diffusion_residual_baseline"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260522
EPOCHS = 26
BATCH_SIZE = 512
T_STEPS = 40
SAMPLE_STEPS = 40
N_SCENARIOS = 96
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
    return [c for c in cols if c in df.columns]


class DiffusionMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 192, time_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
        )
        self.net = nn.Sequential(
            nn.Linear(n_features + 1 + hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.time_dim = time_dim

    def sinusoidal_t(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), half, device=t.device))
        args = t.float().unsqueeze(1) / freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(self.sinusoidal_t(t))
        z = torch.cat([x_t.view(-1, 1), cond, emb], dim=1)
        return self.net(z).squeeze(-1)


def diffusion_schedule() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1e-4, 0.04, T_STEPS, device=DEVICE)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def train_diffusion(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> DiffusionMLP:
    seed_all(seed)
    beta, alpha, alpha_bar = diffusion_schedule()
    model = DiffusionMLP(x_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            t = torch.randint(0, T_STEPS, (len(yb),), device=DEVICE)
            eps = torch.randn_like(yb)
            ab = alpha_bar[t]
            x_t = torch.sqrt(ab) * yb + torch.sqrt(1.0 - ab) * eps
            pred = model(x_t, t, xb)
            loss = nn.functional.mse_loss(pred, eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def sample_scenarios(model: DiffusionMLP, x_cond: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    beta, alpha, alpha_bar = diffusion_schedule()
    model.eval()
    scenarios = []
    with torch.no_grad():
        for start in range(0, len(x_cond), 512):
            cond = torch.tensor(x_cond[start : start + 512], dtype=torch.float32, device=DEVICE)
            n = cond.shape[0]
            cond_rep = cond.repeat_interleave(N_SCENARIOS, dim=0)
            x_t = torch.randn(n * N_SCENARIOS, device=DEVICE)
            for step in reversed(range(SAMPLE_STEPS)):
                t = torch.full((n * N_SCENARIOS,), step, device=DEVICE, dtype=torch.long)
                eps = model(x_t, t, cond_rep)
                a = alpha[step]
                ab = alpha_bar[step]
                b = beta[step]
                mean = (x_t - (b / torch.sqrt(1.0 - ab)) * eps) / torch.sqrt(a)
                if step > 0:
                    x_t = mean + torch.sqrt(b) * torch.randn_like(x_t)
                else:
                    x_t = mean
            arr = x_t.view(n, N_SCENARIOS).cpu().numpy() * y_std + y_mean
            scenarios.append(arr)
    return np.vstack(scenarios)


def fit_fold(long_df: pd.DataFrame, eval_df: pd.DataFrame, month: str, features: list[str]) -> pd.DataFrame:
    test_times = set(eval_df.loc[eval_df["test_month"].eq(month), "time"])
    test = long_df[long_df["time"].isin(test_times)].copy().reset_index(drop=True)
    test_start = pd.Timestamp(f"{month}-01")
    train = long_df[long_df["time"] < test_start].copy().reset_index(drop=True)
    tau = float(eval_df.loc[eval_df["test_month"].eq(month), "tau_negative"].iloc[0])
    med = train[features].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    scaler = StandardScaler()
    train_x0 = train[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    scaler.fit(train_x0)

    def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = scaler.transform(frame[features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)).astype(np.float32)
        y = frame["spread_rt_minus_da"].to_numpy(dtype=np.float32)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        return x[valid], y[valid]

    x_train, y_train_raw = xy(train)
    x_test, _ = xy(test)
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw) + 1e-6)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
    out = test[["time"]].copy()
    out["test_month"] = month
    for col in ["c33_diff_tail_prob", "c33_diff_expected_shortfall", "c33_diff_q05_risk", "c33_diff_q10_risk", "c33_diff_q20_risk", "c33_diff_scenario_std"]:
        out[col] = np.nan
    if len(x_train) < 1000 or len(x_test) == 0:
        return out
    model = train_diffusion(x_train, y_train, SEED)
    scen = sample_scenarios(model, x_test, y_mean, y_std)
    out["c33_diff_tail_prob"] = (scen < tau).mean(axis=1)
    out["c33_diff_expected_shortfall"] = np.maximum(tau - scen, 0).mean(axis=1)
    out["c33_diff_q05_risk"] = tau - np.quantile(scen, 0.05, axis=1)
    out["c33_diff_q10_risk"] = tau - np.quantile(scen, 0.10, axis=1)
    out["c33_diff_q20_risk"] = tau - np.quantile(scen, 0.20, axis=1)
    out["c33_diff_scenario_std"] = scen.std(axis=1)
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
    long_df = add_features(build_long_feature_panel())
    features = feature_cols(long_df)
    parts = [fit_fold(long_df, eval_df, month, features) for month in sorted(eval_df["test_month"].unique())]
    pred = pd.concat(parts, ignore_index=True)
    c20_full = pd.read_csv(C20_FULL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    full = c20_full.merge(pred.drop(columns=["test_month"]), on="time", how="left")
    base = build_base_score(full)
    for col in ["c33_diff_tail_prob", "c33_diff_expected_shortfall", "c33_diff_q20_risk"]:
        r = full[col].rank(method="average", pct=True)
        full[f"c33_blend_{col}_base_w0p2"] = 0.2 * r + 0.8 * base
        full[f"c33_blend_{col}_base_w0p35"] = 0.35 * r + 0.65 * base
    score_cols = [
        "c33_diff_tail_prob",
        "c33_diff_expected_shortfall",
        "c33_diff_q05_risk",
        "c33_diff_q10_risk",
        "c33_diff_q20_risk",
        "c33_diff_scenario_std",
    ] + [c for c in full.columns if c.startswith("c33_blend_")]
    metrics = fixed_budget_metrics(full, score_cols)
    full.to_csv(OUT / "c33_conditional_diffusion_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c33_conditional_diffusion_metrics.csv", index=False, encoding="utf-8-sig")
    pressure30 = metrics[metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    report = [
        "# C33 Conditional Diffusion Residual/Scenario Baseline",
        "",
        "Purpose: add a generative probabilistic forecasting baseline. The model samples conditional spread scenarios and derives tail probability, expected shortfall, and lower-quantile risks.",
        "",
        f"Device: `{DEVICE}`",
        "",
        f"Features: {len(features)}; diffusion steps: {T_STEPS}; scenarios per timestamp: {N_SCENARIOS}.",
        "",
        "## Pressure 30% Fixed-Budget Results",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## All-Month 30% Fixed-Budget Results",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
    ]
    (OUT / "c33_conditional_diffusion_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
