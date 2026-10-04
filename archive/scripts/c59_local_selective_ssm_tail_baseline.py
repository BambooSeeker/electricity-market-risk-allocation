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
OUT = ROOT / "c59_local_selective_ssm_tail_baseline"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260523
SEQ_LEN = 96
EPOCHS = 8
BATCH_SIZE = 512
ENSEMBLE_SEEDS = [0]
QUANTILES = np.array([0.05, 0.10, 0.20, 0.50], dtype=np.float32)
BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SelectiveSSMLayer(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.candidate = nn.Linear(n_features, hidden)
        self.input_gate = nn.Linear(n_features, hidden)
        self.decay_gate = nn.Linear(n_features, hidden)
        self.skip = nn.Linear(n_features, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.candidate.out_features)
        outs = []
        cand = torch.tanh(self.candidate(x))
        inp = torch.sigmoid(self.input_gate(x))
        decay = torch.sigmoid(self.decay_gate(x))
        skip = self.skip(x)
        for t in range(steps):
            proposal = inp[:, t, :] * cand[:, t, :] + 0.15 * skip[:, t, :]
            h = decay[:, t, :] * h + (1.0 - decay[:, t, :]) * proposal
            outs.append(h)
        return self.norm(torch.stack(outs, dim=1))


class SelectiveSSMRiskNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.ssm1 = SelectiveSSMLayer(n_features, hidden)
        self.mix1 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.ssm2 = SelectiveSSMLayer(hidden, hidden)
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.quant = nn.Linear(hidden // 2, len(QUANTILES))
        self.cls = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        h = self.ssm1(x)
        h = self.ssm2(self.mix1(h))
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        z = torch.sum(h * weights.unsqueeze(-1), dim=1)
        z = self.head(z)
        q = torch.sort(self.quant(z), dim=1).values
        logit = self.cls(z).squeeze(-1)
        return q, logit


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor(QUANTILES, device=q_pred.device).view(1, -1)
    err = y.view(-1, 1) - q_pred
    return torch.maximum(qs * err, (qs - 1.0) * err).mean()


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


def train_one(x_train: np.ndarray, y_train: np.ndarray, tau: float, seed: int) -> SelectiveSSMRiskNet:
    seed_all(seed)
    model = SelectiveSSMRiskNet(x_train.shape[-1]).to(DEVICE)
    y_tail = (y_train < tau).astype(np.float32)
    pos = y_tail.sum()
    pos_weight = torch.tensor([(len(y_tail) - pos) / max(pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_tail, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            q, logit = model(xb)
            loss = pinball_loss(q, yb) + 0.45 * bce(logit, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict(models: list[SelectiveSSMRiskNet], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    print(f"[C59] fold {month} start", flush=True)
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
    for col in [
        "c59_selssm_tail_prob",
        "c59_selssm_tail_prob_iso",
        "c59_selssm_q05_risk",
        "c59_selssm_q10_risk",
        "c59_selssm_q20_risk",
        "c59_selssm_q50_risk",
        "c59_selssm_expected_shortfall",
    ]:
        out[col] = np.nan
    if len(x_train) < 1000 or len(x_test) == 0:
        print(f"[C59] fold {month} skipped: train={len(x_train)} test={len(x_test)}", flush=True)
        return out

    print(f"[C59] fold {month} train={len(x_train)} cal={len(x_cal)} test={len(x_test)}", flush=True)
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

    out.loc[pos_test, "c59_selssm_tail_prob"] = p_test
    out.loc[pos_test, "c59_selssm_tail_prob_iso"] = p_iso
    out.loc[pos_test, "c59_selssm_q05_risk"] = tau - q_test[:, 0]
    out.loc[pos_test, "c59_selssm_q10_risk"] = tau - q_test[:, 1]
    out.loc[pos_test, "c59_selssm_q20_risk"] = tau - q_test[:, 2]
    out.loc[pos_test, "c59_selssm_q50_risk"] = tau - q_test[:, 3]
    out.loc[pos_test, "c59_selssm_expected_shortfall"] = np.maximum(tau - q_test[:, :3], 0).mean(axis=1)
    print(f"[C59] fold {month} done", flush=True)
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
        "c59_selssm_tail_prob_iso",
        "c59_selssm_q20_risk",
        "c59_selssm_expected_shortfall",
    ]
    for col in rank_sources:
        r = monthly_rank(full, col).fillna(0.5)
        full[f"c59_blend_{col}_base_w0p2"] = 0.8 * full["base_score"] + 0.2 * r
        full[f"c59_blend_{col}_c22_w0p2"] = full["c22_boundary_score"] + 0.2 * r
    full["c59_selssm_consensus"] = full[[f"c59_blend_{col}_base_w0p2" for col in rank_sources]].mean(axis=1)
    full["c59_c22_selssm_consensus"] = full[[f"c59_blend_{col}_c22_w0p2" for col in rank_sources]].mean(axis=1)
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
        "c59_selssm_tail_prob",
        "c59_selssm_tail_prob_iso",
        "c59_selssm_q05_risk",
        "c59_selssm_q10_risk",
        "c59_selssm_q20_risk",
        "c59_selssm_q50_risk",
        "c59_selssm_expected_shortfall",
        "c59_selssm_consensus",
        "c59_c22_selssm_consensus",
    ] + [c for c in full.columns if c.startswith("c59_blend_")]
    metrics = fixed_budget_metrics(full, score_cols)

    full.to_csv(OUT / "c59_local_selective_ssm_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c59_local_selective_ssm_metrics.csv", index=False, encoding="utf-8-sig")

    pressure30 = metrics[metrics["scope"].eq("pressure_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    all30 = metrics[metrics["scope"].eq("all_available_months") & metrics["budget"].eq(0.30)].sort_values("gate_score", ascending=False)
    report = [
        "# C59 Local Selective-SSM Tail Baseline",
        "",
        "Purpose: run a local GPU stronger modern sequence baseline after A800 is unavailable. This is a selective-SSM-style supervised baseline, not a claim of using official Mamba/S4.",
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
        "## Interpretation",
        "",
        "- If this baseline improves C50 or C44 under the frozen metrics, it can strengthen the modern-method line.",
        "- If it does not, it remains a stronger local modern baseline check and should not alter the main claim.",
    ]
    (OUT / "c59_local_selective_ssm_tail_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"device={DEVICE} features={len(features)}")
    print("pressure30")
    print(pressure30[["method", "recall", "excess_share", "false_alert_burden", "gate_score"]].head(10).to_string(index=False))
    print("all30")
    print(all30[["method", "recall", "excess_share", "false_alert_burden", "gate_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
