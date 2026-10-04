from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_family_complementarity_audit" / "merged_family_scores.csv"
SLTS = ROOT / "slts_canonical_outputs" / "predictions_all_methods.csv"
OUT = ROOT / "c2_deep_severity_sequence_model"
OUT.mkdir(exist_ok=True)

SEED = 20260520
PRESSURE_MONTHS = {"2025-09", "2025-10"}
BUDGETS = [0.10, 0.20, 0.30, 0.40]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEATURES = [
    "QIA_two_stage",
    "HGB_q10_distance_two_stage",
    "LightGBM_q10_distance_two_stage",
    "TreeML_rank_avg",
    "AllSignals_equal_rank_avg",
    "DeepRank_neg_risk_wide_g12",
    "qia_component_raw_negative",
    "qia_component_tail_regime",
    "qia_component_signed_direction",
    "q05",
    "q10",
    "q50",
    "q_width_90",
    "neg_distance_q05",
    "neg_distance_q10",
    "abs_distance",
    "hour_sin",
    "hour_cos",
    "is_midday",
    "is_morning_ramp",
    "q_width_x_midday",
    "q_width_x_morning",
    "q10_x_midday",
    "q10_x_morning",
]


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def load_panel() -> pd.DataFrame:
    base = pd.read_csv(BASE, parse_dates=["time"])
    qia = pd.read_csv(SLTS, parse_dates=["time"])
    qia = qia[qia["method"] == "QIA_two_stage"].copy()
    qia = qia[
        [
            "fold",
            "test_month",
            "time",
            "component_raw_negative",
            "component_tail_regime",
            "component_signed_direction",
            "q05",
            "q10",
            "q50",
            "q_width_90",
            "neg_distance_q05",
            "neg_distance_q10",
            "abs_distance",
        ]
    ].rename(
        columns={
            "component_raw_negative": "qia_component_raw_negative",
            "component_tail_regime": "qia_component_tail_regime",
            "component_signed_direction": "qia_component_signed_direction",
        }
    )
    qia["_has_quantiles"] = qia["q_width_90"].notna().astype(int)
    qia = (
        qia.sort_values(["fold", "test_month", "time", "_has_quantiles"], ascending=[True, True, True, False])
        .drop_duplicates(["fold", "test_month", "time"], keep="first")
        .drop(columns=["_has_quantiles"])
    )
    df = base.merge(qia, on=["fold", "test_month", "time"], how="left")
    if df.duplicated(["fold", "test_month", "time"]).any():
        raise ValueError("load_panel produced duplicate fold/test_month/time keys")
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("time").reset_index(drop=True)
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_midday"] = ((hour >= 11) & (hour <= 15.5)).astype(float)
    df["is_morning_ramp"] = ((hour >= 6) & (hour <= 10.5)).astype(float)
    df["q_width_x_midday"] = df["q_width_90"] * df["is_midday"]
    df["q_width_x_morning"] = df["q_width_90"] * df["is_morning_ramp"]
    df["q10_x_midday"] = df["q10"] * df["is_midday"]
    df["q10_x_morning"] = df["q10"] * df["is_morning_ramp"]
    for col in FEATURES + ["negative_tail", "negative_excess"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, excess: np.ndarray, seq_len: int):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.excess = excess.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        start = max(0, idx - self.seq_len + 1)
        seq = self.x[start : idx + 1]
        if len(seq) < self.seq_len:
            pad = np.repeat(seq[:1], self.seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        return torch.from_numpy(seq), torch.tensor(self.y[idx]), torch.tensor(self.excess[idx])


class LSTMAttn(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class GRUAttn(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class TinyTCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        layers = []
        in_ch = n_features
        for dilation in [1, 2, 4, 8]:
            layers += [
                nn.Conv1d(in_ch, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
            ]
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        z = x.transpose(1, 2)
        return self.head(self.net(z)).squeeze(-1)


@dataclass
class Config:
    name: str
    model_type: str
    seq_len: int
    hidden: int
    layers: int
    dropout: float
    lr: float
    severity_weight: float
    epochs: int = 45


CONFIGS = [
    Config("lstm_s48_h64_w4", "lstm", 48, 64, 2, 0.15, 1e-3, 4.0),
    Config("lstm_s96_h96_w6", "lstm", 96, 96, 2, 0.20, 8e-4, 6.0),
    Config("gru_s48_h64_w6", "gru", 48, 64, 2, 0.15, 1e-3, 6.0),
    Config("gru_s96_h96_w8", "gru", 96, 96, 2, 0.20, 8e-4, 8.0),
    Config("tcn_s48_h64_w6", "tcn", 48, 64, 1, 0.15, 1e-3, 6.0),
    Config("tcn_s96_h96_w8", "tcn", 96, 96, 1, 0.20, 8e-4, 8.0),
]


def make_model(cfg: Config, n_features: int) -> nn.Module:
    if cfg.model_type == "lstm":
        return LSTMAttn(n_features, cfg.hidden, cfg.layers, cfg.dropout)
    if cfg.model_type == "gru":
        return GRUAttn(n_features, cfg.hidden, cfg.layers, cfg.dropout)
    if cfg.model_type == "tcn":
        return TinyTCN(n_features, cfg.hidden, cfg.dropout)
    raise ValueError(cfg.model_type)


def train_predict_month(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].to_numpy())
    x_test = scaler.transform(test[FEATURES].to_numpy())
    y_train = train["negative_tail"].astype(int).to_numpy()
    excess_train = train["negative_excess"].astype(float).to_numpy()
    ds = SeqDataset(x_train, y_train, excess_train, cfg.seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    model = make_model(cfg, len(FEATURES)).to(DEVICE)
    pos_rate = max(y_train.mean(), 1e-4)
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate, device=DEVICE, dtype=torch.float32).clamp(1, 20)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    best_state = None
    best_loss = math.inf
    for _ in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(DEVICE), yb.to(DEVICE), eb.to(DEVICE)
            logits = model(xb)
            base = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            sev = 1.0 + cfg.severity_weight * torch.clamp(torch.log1p(eb) / 5.0, 0, 4)
            loss = (base * sev).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    ds_test = SeqDataset(x_test, np.zeros(len(x_test)), np.zeros(len(x_test)), cfg.seq_len)
    pred = []
    with torch.no_grad():
        for xb, _, _ in DataLoader(ds_test, batch_size=512, shuffle=False):
            pred.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(pred)


def evaluate_budget(df: pd.DataFrame, score_col: str, budget: float) -> dict[str, float]:
    test = df.sort_values(score_col, ascending=False).copy()
    n_alert = max(1, int(np.ceil(len(test) * budget)))
    alert = np.zeros(len(test), dtype=bool)
    alert[:n_alert] = True
    y = test["negative_tail"].astype(int).to_numpy()
    excess = test["negative_excess"].astype(float).to_numpy()
    hit = y[alert].sum()
    events = y.sum()
    alerts = alert.sum()
    precision = hit / alerts if alerts else np.nan
    recall = hit / events if events else np.nan
    alert_excess = excess[alert].sum()
    total_excess = excess.sum()
    return {
        "n": len(test),
        "events": int(events),
        "precision": float(precision),
        "recall": float(recall),
        "excess_share": float(alert_excess / max(total_excess, 1e-9)),
        "missed_event_share": float(1 - recall),
        "missed_excess_share": float(1 - alert_excess / max(total_excess, 1e-9)),
        "false_alert_burden": float(1 - precision),
    }


def evaluate_ranking(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = df["negative_tail"].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    return {
        "auc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan,
    }


def summarize(panel: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = {
        "all_available_months": panel,
        "pressure_available_months": panel[panel["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": panel[~panel["test_month"].isin(PRESSURE_MONTHS)],
    }
    budget_rows, rank_rows = [], []
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for col in score_cols:
            r = {"scope": scope, "method": col, "n_months": sdf["test_month"].nunique()}
            r.update(evaluate_ranking(sdf, col))
            rank_rows.append(r)
            for budget in BUDGETS:
                item = {"scope": scope, "method": col, "budget": budget}
                item.update(evaluate_budget(sdf, col, budget))
                budget_rows.append(item)
    return pd.DataFrame(budget_rows), pd.DataFrame(rank_rows)


def main() -> None:
    df = load_panel()
    months = sorted(df["test_month"].unique())
    outputs = []
    for cfg in CONFIGS:
        cfg_outputs = []
        for month in months:
            train = df[df["test_month"] < month].copy()
            test = df[df["test_month"] == month].copy()
            if train["negative_tail"].nunique() < 2 or train.empty:
                continue
            pred = train_predict_month(train, test, cfg)
            part = test[["fold", "test_month", "time", "spread_rt_minus_da", "negative_tail", "negative_excess", "tau_negative"]].copy()
            part[cfg.name] = pred
            cfg_outputs.append(part)
        outputs.append(pd.concat(cfg_outputs, ignore_index=True))

    # Merge all config predictions on common available months/rows.
    panel = outputs[0]
    for part in outputs[1:]:
        panel = panel.merge(part[["fold", "test_month", "time", part.columns[-1]]], on=["fold", "test_month", "time"], how="inner")
    base_cols = ["fold", "test_month", "time", "QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]
    panel = panel.merge(df[base_cols], on=["fold", "test_month", "time"], how="left")

    # Rank blends with strongest baseline.
    for cfg in CONFIGS:
        panel[f"{cfg.name}_qia_blend"] = 0.65 * rank01(panel["QIA_two_stage"].to_numpy()) + 0.35 * rank01(panel[cfg.name].to_numpy())
        panel[f"{cfg.name}_hgb_blend"] = 0.65 * rank01(panel["HGB_q10_distance_two_stage"].to_numpy()) + 0.35 * rank01(panel[cfg.name].to_numpy())

    score_cols = ["QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]
    for cfg in CONFIGS:
        score_cols += [cfg.name, f"{cfg.name}_qia_blend", f"{cfg.name}_hgb_blend"]

    budget, ranking = summarize(panel, score_cols)
    panel.to_csv(OUT / "c2_deep_sequence_predictions.csv", index=False, encoding="utf-8-sig")
    budget.to_csv(OUT / "c2_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT / "c2_ranking_metrics.csv", index=False, encoding="utf-8-sig")

    key = budget[
        budget["budget"].isin([0.20, 0.30])
        & budget["scope"].isin(["all_available_months", "pressure_available_months"])
    ].copy()
    key = key.sort_values(["scope", "budget", "excess_share", "recall"], ascending=[True, True, False, False])
    key.to_csv(OUT / "c2_key_budget_rows.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = [
        "# C2 Deep Severity-Aware Sequence Model",
        "",
        f"Device: {DEVICE}",
        "",
        "## Best Key Rows by Scope/Budget",
        "",
        md_table(key, ["scope", "method", "budget", "recall", "excess_share", "missed_event_share", "missed_excess_share", "false_alert_burden"], n=40),
        "",
        "## Gate",
        "",
        "Keep only models that improve pressure-month or event-weighted recall/excess without materially worsening false-alert burden.",
        "",
    ]
    (OUT / "c2_deep_sequence_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
