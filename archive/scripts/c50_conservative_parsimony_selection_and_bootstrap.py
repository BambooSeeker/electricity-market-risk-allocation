from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c9_pressure_miss_diagnosis_and_conditioned_adjustment import metric_from_mask
from c49_pressure_transfer_safe_candidate_selection import OUT as C49_OUT
from c49_pressure_transfer_safe_candidate_selection import PRED, candidate_alert, evaluate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c50_conservative_parsimony_selection_and_bootstrap"
OUT.mkdir(exist_ok=True)

REF = "C44_limited_budget_band_until30"
BOOT_N = 2000
RNG = np.random.default_rng(20260523)


def parse_weight(method: str) -> float:
    if "protected_core" in method:
        return 0.0
    for token, weight in [("w10", 0.10), ("w20", 0.20), ("w30", 0.30)]:
        if token in method:
            return weight
    return 1.0


def family_priority(method: str) -> int:
    if "_deep_" in method:
        return 0
    if "_modern_" in method:
        return 1
    if "_diffusion_" in method:
        return 2
    if "protected_core" in method:
        return 3
    return 4


def conservative_select() -> tuple[str, pd.DataFrame]:
    sel = pd.read_csv(C49_OUT / "c49_pressure_transfer_safe_selection.csv")
    best_val = sel["val_tight_mean_gate_delta"].max()
    sel["modern_weight"] = sel["method"].map(parse_weight)
    sel["family_priority"] = sel["method"].map(family_priority)
    sel["passes_conservative_pool"] = (
        (sel["val_tight_mean_gate_delta"] >= 0.85 * best_val)
        & (sel["val_tight_min_gate_delta"] >= 0)
        & (sel["val_tight_min_recall_delta"] >= 0)
        & (sel["val_tight_min_excess_delta"] >= 0)
        & (sel["val_broad_min_gate_delta"] >= 0)
        & (sel["val_broad_min_recall_delta"] >= 0)
        & (sel["val_broad_min_excess_delta"] >= 0)
        & (sel["max_boundary_displacement_gap"] <= 0)
        & (sel["max_high_boundary_displacement_gap"] <= 0)
    )
    pool = sel[sel["passes_conservative_pool"]].copy()
    if pool.empty:
        selected = REF
    else:
        pool = pool.sort_values(
            [
                "modern_weight",
                "family_priority",
                "max_high_boundary_displacement_gap",
                "val_tight_mean_gate_delta",
            ],
            ascending=[True, True, True, False],
        )
        selected = str(pool.iloc[0]["method"])
    sel["selected_by_conservative_parsimony_rule"] = sel["method"].eq(selected)
    sel["selection_note"] = (
        "85pct_validation_gain_pool_then_lowest_modern_weight_then_single_family_then_boundary_protection"
    )
    return selected, sel.sort_values(
        ["selected_by_conservative_parsimony_rule", "passes_conservative_pool", "modern_weight", "family_priority"],
        ascending=[False, False, True, True],
    )


def metric_from_indices(indices: np.ndarray, alert_full: np.ndarray, y_full: np.ndarray, excess_full: np.ndarray) -> dict:
    y = y_full[indices]
    excess = excess_full[indices]
    alert = alert_full[indices].astype(bool)
    return metric_from_mask(pd.DataFrame({"negative_tail": y, "negative_excess": excess}), alert)


def bootstrap_delta(df: pd.DataFrame, selected: str, n_iter: int = BOOT_N) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["time"].dt.date.astype(str)
    y_full = df["negative_tail"].astype(int).to_numpy()
    excess_full = df["negative_excess"].astype(float).to_numpy()
    rows = []
    scopes = {
        "all_available_months": df,
        "pressure_available_months": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }
    all_alerts = {}
    for budget in [0.25, 0.30]:
        all_alerts[(budget, REF)] = candidate_alert(df, REF, budget).to_numpy(dtype=bool)
        all_alerts[(budget, selected)] = candidate_alert(df, selected, budget).to_numpy(dtype=bool)
    for boot_type in ["day_block", "month_block"]:
        for scope, sdf in scopes.items():
            if boot_type == "day_block":
                groups = [g.index.to_numpy(dtype=int) for _, g in sdf.groupby("date", sort=True)]
            else:
                groups = [g.index.to_numpy(dtype=int) for _, g in sdf.groupby("test_month", sort=True)]
            n_groups = len(groups)
            for i in range(n_iter):
                sampled = RNG.integers(0, n_groups, size=n_groups)
                idx = np.concatenate([groups[j] for j in sampled])
                for budget in [0.25, 0.30]:
                    ref = metric_from_indices(idx, all_alerts[(budget, REF)], y_full, excess_full)
                    tgt = metric_from_indices(idx, all_alerts[(budget, selected)], y_full, excess_full)
                    ref_gate = ref["recall"] + ref["excess_share"] - 0.2 * ref["false_alert_burden"]
                    tgt_gate = tgt["recall"] + tgt["excess_share"] - 0.2 * tgt["false_alert_burden"]
                    rows.append(
                        {
                            "bootstrap_type": boot_type,
                            "scope": scope,
                            "budget": budget,
                            "delta_recall": tgt["recall"] - ref["recall"],
                            "delta_excess_share": tgt["excess_share"] - ref["excess_share"],
                            "delta_gate_score": tgt_gate - ref_gate,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_boot(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in samples.groupby(["bootstrap_type", "scope", "budget"]):
        row = dict(zip(["bootstrap_type", "scope", "budget"], keys))
        for col in ["delta_recall", "delta_excess_share", "delta_gate_score"]:
            vals = g[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_p025"] = float(np.percentile(vals, 2.5))
            row[f"{col}_p975"] = float(np.percentile(vals, 97.5))
        row["p_delta_gate_positive"] = float((g["delta_gate_score"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    selected, selection = conservative_select()
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    metrics = evaluate(df)
    delta = pd.read_csv(C49_OUT / "c49_delta_vs_c44.csv")
    selected_delta = delta[delta["method"].eq(selected)].sort_values(["scope", "budget"])
    samples = bootstrap_delta(df, selected)
    boot = summarize_boot(samples)

    selection.to_csv(OUT / "c50_conservative_selection.csv", index=False, encoding="utf-8-sig")
    selected_delta.to_csv(OUT / "c50_selected_delta_vs_c44.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT / "c50_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "c50_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    focus_boot = boot[boot["scope"].isin(["all_available_months", "pressure_available_months"])].sort_values(
        ["bootstrap_type", "scope", "budget"]
    )
    report = [
        "# C50 Conservative Parsimony Selection and Bootstrap",
        "",
        "Purpose: C49's validation-best rule still selected a holdout-weak candidate. C50 uses a more conservative rule: among candidates with at least 85% of the best validation gain, choose lower modern weight, single-family complement, and stronger boundary protection.",
        "",
        f"Selected method: `{selected}`.",
        "",
        "## Selection",
        "",
        md_table(
            selection,
            [
                "method",
                "passes_conservative_pool",
                "selected_by_conservative_parsimony_rule",
                "val_tight_mean_gate_delta",
                "modern_weight",
                "family_priority",
                "max_boundary_displacement_gap",
                "max_high_boundary_displacement_gap",
            ],
            30,
        ),
        "",
        "## Selected Delta vs C44",
        "",
        md_table(
            selected_delta,
            ["scope", "budget", "recall", "excess_share", "false_alert_burden", "gate_score", "delta_recall", "delta_excess_share", "delta_gate_score"],
            20,
        ),
        "",
        "## Bootstrap Summary",
        "",
        md_table(
            focus_boot,
            [
                "bootstrap_type",
                "scope",
                "budget",
                "delta_recall_mean",
                "delta_recall_p025",
                "delta_recall_p975",
                "delta_excess_share_mean",
                "delta_excess_share_p025",
                "delta_excess_share_p975",
                "delta_gate_score_mean",
                "delta_gate_score_p025",
                "delta_gate_score_p975",
                "p_delta_gate_positive",
            ],
            20,
        ),
        "",
        "## Interpretation",
        "",
        "- This rule is deliberately conservative; it gives up some validation gain to reduce transfer risk.",
        "- If bootstrap remains weak, the modern complement should still be framed cautiously as a robustness-enhancing candidate, not a settled main method.",
    ]
    (OUT / "c50_conservative_parsimony_selection_and_bootstrap_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
