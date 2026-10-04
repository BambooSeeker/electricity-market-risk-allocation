from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS, rank01


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "c2_deep_severity_sequence_model" / "c2_deep_sequence_predictions.csv"
OUT = ROOT / "c3_validation_safe_diversified_policy"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.20, 0.30]
BASES = ["QIA_two_stage", "HGB_q10_distance_two_stage", "AllSignals_equal_rank_avg"]
DEEPS = ["lstm_s96_h96_w6", "gru_s48_h64_w6", "tcn_s48_h64_w6", "tcn_s96_h96_w8"]
SHARES = [0.35, 0.45, 0.55, 0.65, 0.75]

CANDIDATE_TUPLES = [
    ("HGB_q10_distance_two_stage", "lstm_s96_h96_w6", 0.15, "tcn_s96_h96_w8", 0.20),
    ("HGB_q10_distance_two_stage", "lstm_s96_h96_w6", 0.20, "gru_s48_h64_w6", 0.35),
    ("HGB_q10_distance_two_stage", "lstm_s96_h96_w6", 0.25, "tcn_s48_h64_w6", 0.30),
    ("HGB_q10_distance_two_stage", "lstm_s96_h96_w6", 0.15, "gru_s48_h64_w6", 0.40),
    ("HGB_q10_distance_two_stage", "lstm_s96_h96_w6", 0.25, "tcn_s48_h64_w6", 0.25),
    ("HGB_q10_distance_two_stage", "tcn_s48_h64_w6", 0.15, "tcn_s96_h96_w8", 0.30),
    ("QIA_two_stage", "lstm_s96_h96_w6", 0.15, "tcn_s96_h96_w8", 0.20),
    ("QIA_two_stage", "lstm_s96_h96_w6", 0.20, "gru_s48_h64_w6", 0.35),
    ("AllSignals_equal_rank_avg", "lstm_s96_h96_w6", 0.20, "gru_s48_h64_w6", 0.35),
    ("AllSignals_equal_rank_avg", "lstm_s96_h96_w6", 0.15, "gru_s48_h64_w6", 0.40),
]


def metric_from_mask(df: pd.DataFrame, alert: np.ndarray) -> dict[str, float]:
    y = df["negative_tail"].astype(int).to_numpy()
    excess = df["negative_excess"].astype(float).to_numpy()
    hit = y[alert].sum()
    events = y.sum()
    alerts = alert.sum()
    precision = hit / alerts if alerts else np.nan
    recall = hit / events if events else np.nan
    alert_excess = excess[alert].sum()
    total_excess = excess.sum()
    return {
        "n": len(df),
        "events": int(events),
        "precision": float(precision),
        "recall": float(recall),
        "excess_share": float(alert_excess / max(total_excess, 1e-9)),
        "missed_event_share": float(1 - recall),
        "missed_excess_share": float(1 - alert_excess / max(total_excess, 1e-9)),
        "false_alert_burden": float(1 - precision),
    }


def alert_mask(df: pd.DataFrame, score_a: str, score_b: str, budget: float, share_a: float) -> np.ndarray:
    n_alert = max(1, int(np.ceil(len(df) * budget)))
    n_a = int(round(n_alert * share_a))
    order_a = np.argsort(-df[score_a].to_numpy())
    order_b = np.argsort(-df[score_b].to_numpy())
    chosen: list[int] = []
    used = set()
    for pos in order_a:
        if len(chosen) >= n_a:
            break
        chosen.append(pos)
        used.add(pos)
    for pos in order_b:
        if len(chosen) >= n_alert:
            break
        if pos not in used:
            chosen.append(pos)
            used.add(pos)
    mask = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return mask


def add_candidate_scores(df: pd.DataFrame) -> list[str]:
    scores = {col: df[col].to_numpy() for col in BASES}
    for base, d1, w1, d2, w2 in CANDIDATE_TUPLES:
        bw = 1.0 - w1 - w2
        name = f"{base}__{d1}_{int(w1*100)}__{d2}_{int(w2*100)}"
        scores[name] = bw * rank01(df[base].to_numpy()) + w1 * rank01(df[d1].to_numpy()) + w2 * rank01(df[d2].to_numpy())
    bank = pd.DataFrame(scores, index=df.index)
    df[bank.columns] = bank
    return list(bank.columns)


def select_policy(val: pd.DataFrame, score_cols: list[str], budget: float) -> tuple[str, str, float, float]:
    single_rows = []
    for score in score_cols:
        mask = alert_mask(val, score, score, budget, 1.0)
        m = metric_from_mask(val, mask)
        gate = m["recall"] + m["excess_share"] - 0.2 * m["false_alert_burden"]
        single_rows.append((score, gate))
    shortlist = [x[0] for x in sorted(single_rows, key=lambda t: t[1], reverse=True)[:10]]
    best = (shortlist[0], shortlist[0], 1.0, -np.inf)
    for a in shortlist:
        for b in shortlist:
            for share in SHARES:
                mask = alert_mask(val, a, b, budget, share)
                m = metric_from_mask(val, mask)
                # Pre-fixed validation objective: severity capture first, then event recall, with false-alert penalty.
                gate = m["excess_share"] + 0.85 * m["recall"] - 0.2 * m["false_alert_burden"]
                if gate > best[3]:
                    best = (a, b, share, gate)
    return best


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_available_months": result,
        "pressure_available_months": result[result["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": result[~result["test_month"].isin(PRESSURE_MONTHS)],
    }
    rows = []
    for scope, sdf in scopes.items():
        if sdf.empty:
            continue
        for budget in BUDGETS:
            alert = sdf[f"alert_b{int(budget*100)}"].astype(bool).to_numpy()
            m = metric_from_mask(sdf, alert)
            rows.append({"scope": scope, "budget": budget, **m})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    score_cols = add_candidate_scores(df)
    months = sorted(df["test_month"].unique())
    rows = []
    policies = []
    for month in months:
        hist_months = [m for m in months if m < month]
        if len(hist_months) < 1:
            continue
        val_months = hist_months[-2:]
        val = df[df["test_month"].isin(val_months)].reset_index(drop=True)
        test = df[df["test_month"].eq(month)].copy().reset_index(drop=True)
        for budget in BUDGETS:
            a, b, share, gate = select_policy(val, score_cols, budget)
            mask = alert_mask(test, a, b, budget, share)
            test[f"alert_b{int(budget*100)}"] = mask
            policies.append(
                {
                    "test_month": month,
                    "budget": budget,
                    "validation_months": ",".join(val_months),
                    "score_a": a,
                    "score_b": b,
                    "share_a": share,
                    "validation_gate": gate,
                }
            )
        rows.append(test)
    result = pd.concat(rows, ignore_index=True)
    metrics = summarize(result)
    metrics["gate_score"] = metrics["recall"] + metrics["excess_share"] - 0.2 * metrics["false_alert_burden"]
    policies_df = pd.DataFrame(policies)

    result.to_csv(OUT / "c3_validation_safe_alert_predictions.csv", index=False, encoding="utf-8-sig")
    policies_df.to_csv(OUT / "c3_selected_policies_by_month.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c3_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")

    pass_gate = metrics[
        (metrics["scope"].eq("pressure_available_months"))
        & (metrics["budget"].eq(0.30))
        & (metrics["recall"] >= 0.70)
        & (metrics["excess_share"] >= 0.75)
    ]
    pass_gate.to_csv(OUT / "c3_pressure_budget30_pass_gate.csv", index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
        view = frame[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = [
        "# C3 Validation-Safe Diversified Fixed-Budget Policy",
        "",
        "Each test month selects score_a, score_b, and share_a using only the two immediately preceding available months.",
        "",
        "## Fixed-Budget Metrics",
        "",
        md_table(metrics, ["scope", "budget", "recall", "excess_share", "missed_event_share", "missed_excess_share", "false_alert_burden", "gate_score"], 20),
        "",
        "## Selected Policies",
        "",
        md_table(policies_df, ["test_month", "budget", "validation_months", "score_a", "score_b", "share_a", "validation_gate"], 30),
        "",
        "## Gate",
        "",
        "Pressure 30% gate passed." if not pass_gate.empty else "Pressure 30% gate not passed under validation-safe selection.",
    ]
    (OUT / "c3_validation_safe_diversified_policy_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
