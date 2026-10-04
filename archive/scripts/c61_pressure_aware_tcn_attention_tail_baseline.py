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
from c32_ssm_like_risk_sequence_baseline import add_state_features, feature_cols


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c61_pressure_aware_tcn_attention_tail_baseline"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260523
SEQ_LEN = 96
EPOCHS = 14
BATCH_SIZE = 512
ENSEMBLE_SEEDS = [0, 11]
QUANTILES = np.array([0.05, 0.10, 0.20, 0.50], dtype=np.float32)
BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sequences(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_all = df[features].to_numpy(np.float32)
    y_all = df["spread_rt_minus_da"].to_numpy(np.float32)
    valid = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
    xs, ys, pos = [], [], []
    for i in range(SEQ_LEN - 1, len(df)):
        sl = slice(i - SEQ_LEN + 1, i + 1)
        if valid[sl].all():
            xs.append(x_all[sl])
            ys.append(y_all[i])
            pos.append(i)
    if not xs:
        return (
            np.empty((0, SEQ_LEN, len(features)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=int),
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(pos, dtype=int)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        a, gate = z.chunk(2, dim=1)
        z = torch.tanh(a) * torch.sigmoid(gate)
        z = self.mix(self.dropout(z))
        return self.norm(x + z)


class DilatedTCNAttentionRiskNet(nn.Module):
    def __init__(self, n_features: int, channels: int = 96, dropout: float = 0.12):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualDilatedBlock(channels, dilation=d, dropout=dropout) for d in [1, 2, 4, 8, 16]]
        )
        self.local_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(nn.Linear(channels, channels // 2), nn.GELU(), nn.Linear(channels // 2, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quant = nn.Linear(channels, len(QUANTILES))
        self.cls = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x).transpose(1, 2)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.local_head(h).transpose(1, 2)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        attn_pool = torch.sum(h * weights.unsqueeze(-1), dim=1)
        last_pool = h[:, -1, :]
        z = self.head(torch.cat([attn_pool, last_pool], dim=1))
        q = torch.sort(self.quant(z), dim=1).values
        logit = self.cls(z).squeeze(-1)
        return q, logit


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor(QUANTILES, device=q_pred.device).view(1, -1)
    err = y.view(-1, 1) - q_pred
    return torch.maximum(qs * err, (qs - 1.0) * err).mean()


def sample_weights(frame: pd.DataFrame, y: np.ndarray, tau: float, pos: np.ndarray) -> np.ndarray:
    raw = frame.iloc[pos].copy()
    weights = np.ones(len(y), dtype=np.float32)
    tail = y < tau
    severe_tail = y < (tau - 35.0)
    weights[tail] *= 1.75
    weights[severe_tail] *= 1.60
    weights[raw["month"].isin(PRESSURE_MONTHS).to_numpy()] *= 1.35
    excess = np.maximum(tau - y, 0.0)
    if np.nanmax(excess) > 0:
        weights *= 1.0 + 0.50 * (excess / np.nanmax(excess)).astype(np.float32)
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return weights.astype(np.float32)


def train_one(x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, tau: float, seed: int) -> DilatedTCNAttentionRiskNet:
    seed_all(seed)
    model = DilatedTCNAttentionRiskNet(x_train.shape[-1]).to(DEVICE)
    y_tail = (y_train < tau).astype(np.float32)
    pos = y_tail.sum()
    pos_weight = torch.tensor([(len(y_tail) - pos) / max(pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_tail, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb, tb, wb in loader:
            xb, yb, tb, wb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE), wb.to(DEVICE)
            q, logit = model(xb)
            qs = torch.tensor(QUANTILES, device=q.device).view(1, -1)
            err = yb.view(-1, 1) - q
            pin = torch.maximum(qs * err, (qs - 1.0) * err).mean(dim=1)
            cls = nn.functional.binary_cross_entropy_with_logits(logit, tb, pos_weight=pos_weight, reduction="none")
            loss = (wb * (pin + 0.50 * cls)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict(models: list[DilatedTCNAttentionRiskNet], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_parts, p_parts = [], []
    for model in models:
        model.eval()
        qs, ps = [], []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                xb = torch.tensor(x[i : i + 2048], dtype=torch.float32, device=DEVICE)
                q, logit = model(xb)
                qs.append(q.cpu().numpy())
                ps.append(torch.sigmoid(logit).cpu().numpy())
        q_parts.append(np.vstack(qs))
        p_parts.append(np.concatenate(ps))
    return np.mean(q_parts, axis=0), np.mean(p_parts, axis=0)


def fit_fold(long_df: pd.DataFrame, eval_df: pd.DataFrame, month: str, features: list[str]) -> pd.DataFrame:
    print(f"[C61] fold {month} start", flush=True)
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
    x_train, y_train, pos_train = make_sequences(core_p, features)
    x_cal, y_cal, _ = make_sequences(cal_p, features)
    x_test, _, pos_test = make_sequences(test_p, features)

    out = test[["time"]].copy()
    out["test_month"] = month
    for col in [
        "c61_tcn_tail_prob",
        "c61_tcn_tail_prob_iso",
        "c61_tcn_q05_risk",
        "c61_tcn_q10_risk",
        "c61_tcn_q20_risk",
        "c61_tcn_q50_risk",
        "c61_tcn_expected_shortfall",
    ]:
        out[col] = np.nan
    if len(x_train) < 1000 or len(x_test) == 0:
        print(f"[C61] fold {month} skipped: train={len(x_train)} test={len(x_test)}", flush=True)
        return out

    train_weights = sample_weights(core_p, y_train, tau, pos_train)
    print(f"[C61] fold {month} train={len(x_train)} cal={len(x_cal)} test={len(x_test)}", flush=True)
    models = [train_one(x_train, y_train, train_weights, tau, SEED + s) for s in ENSEMBLE_SEEDS]
    q_test, p_test = predict(models, x_test)
    p_iso = p_test.copy()
    if len(x_cal) > 50:
        _, p_cal = predict(models, x_cal)
        y_tail = (y_cal < tau).astype(int)
        if len(np.unique(y_tail)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, y_tail)
            p_iso = iso.predict(p_test)

    out.loc[pos_test, "c61_tcn_tail_prob"] = p_test
    out.loc[pos_test, "c61_tcn_tail_prob_iso"] = p_iso
    out.loc[pos_test, "c61_tcn_q05_risk"] = tau - q_test[:, 0]
    out.loc[pos_test, "c61_tcn_q10_risk"] = tau - q_test[:, 1]
    out.loc[pos_test, "c61_tcn_q20_risk"] = tau - q_test[:, 2]
    out.loc[pos_test, "c61_tcn_q50_risk"] = tau - q_test[:, 3]
    out.loc[pos_test, "c61_tcn_expected_shortfall"] = np.maximum(tau - q_test[:, :3], 0).mean(axis=1)
    print(f"[C61] fold {month} done", flush=True)
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
                if tmp.empty:
                    continue
                tmp = tmp.rename(columns={col: "_score"})
                alert = monthly_alert_mask(tmp, "_score", budget)
                rows.append({"scope": scope, "method": col, "budget": budget, **metric_from_mask(tmp, alert)})
    out = pd.DataFrame(rows)
    out["gate_score"] = out["recall"] + out["excess_share"] - 0.2 * out["false_alert_burden"]
    return out


def add_blends(full: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    full["base_score"] = build_base_score(full)
    full["c22_boundary_score"] = full["base_score"] + 0.2 * monthly_rank(full, BONUS_FEATURE).fillna(0.5)
    rank_sources = [
        "c61_tcn_tail_prob_iso",
        "c61_tcn_q20_risk",
        "c61_tcn_expected_shortfall",
    ]
    for col in rank_sources:
        r = monthly_rank(full, col).fillna(0.5)
        full[f"c61_blend_{col}_base_w0p2"] = 0.8 * full["base_score"] + 0.2 * r
        full[f"c61_blend_{col}_c22_w0p2"] = full["c22_boundary_score"] + 0.2 * r
    full["c61_tcn_consensus"] = full[[f"c61_blend_{col}_base_w0p2" for col in rank_sources]].mean(axis=1)
    full["c61_c22_tcn_consensus"] = full[[f"c61_blend_{col}_c22_w0p2" for col in rank_sources]].mean(axis=1)
    return full


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    seed_all(SEED)
    eval_df = pd.read_csv(EVAL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    long_df = add_state_features(build_long_feature_panel())
    features = feature_cols(long_df)
    parts = [fit_fold(long_df, eval_df, month, features) for month in sorted(eval_df["test_month"].unique())]
    pred = pd.concat(parts, ignore_index=True)
    base = pd.read_csv(C20_FULL_PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    full = base.merge(pred.drop(columns=["test_month"]), on="time", how="left")
    full = add_blends(full)

    score_cols = [
        "c61_tcn_tail_prob",
        "c61_tcn_tail_prob_iso",
        "c61_tcn_q05_risk",
        "c61_tcn_q10_risk",
        "c61_tcn_q20_risk",
        "c61_tcn_q50_risk",
        "c61_tcn_expected_shortfall",
        "c61_tcn_consensus",
        "c61_c22_tcn_consensus",
    ] + [c for c in full.columns if c.startswith("c61_blend_")]
    metrics = fixed_budget_metrics(full, score_cols)

    full.to_csv(OUT / "c61_pressure_aware_tcn_attention_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c61_pressure_aware_tcn_attention_metrics.csv", index=False, encoding="utf-8-sig")

    pressure30 = metrics[metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    budget_best = metrics.sort_values("gate_score", ascending=False).groupby(["scope", "budget"], as_index=False).head(1)
    report = [
        "# C61 Pressure-Aware TCN Attention Tail Baseline",
        "",
        "Purpose: test whether pressure-aware sample weighting can improve the C60 multi-scale TCN-attention risk model under the frozen fixed-budget lower-tail risk coverage metrics.",
        "",
        f"Device: `{DEVICE}`; sequence length: {SEQ_LEN}; ensemble seeds: {len(ENSEMBLE_SEEDS)}; epochs: {EPOCHS}; features: {len(features)}.",
        "",
        "## Pressure 30% Fixed-Budget Results",
        "",
        md_table(pressure30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## All-Month 30% Fixed-Budget Results",
        "",
        md_table(all30, ["method", "recall", "excess_share", "false_alert_burden", "gate_score"], 25),
        "",
        "## Best Result by Scope and Budget",
        "",
        md_table(budget_best.sort_values(["scope", "budget"]), ["scope", "budget", "method", "recall", "excess_share", "gate_score"], 30),
    ]
    (OUT / "c61_pressure_aware_tcn_attention_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"device={DEVICE} features={len(features)}")
    print("pressure30")
    print(pressure30[["method", "recall", "excess_share", "false_alert_burden", "gate_score"]].head(10).to_string(index=False))
    print("all30")
    print(all30[["method", "recall", "excess_share", "false_alert_burden", "gate_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

