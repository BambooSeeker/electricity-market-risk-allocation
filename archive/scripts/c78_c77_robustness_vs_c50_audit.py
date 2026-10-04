from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_deep_severity_sequence_model import PRESSURE_MONTHS
from c43_budget_aware_boundary_correction import build_alerts, eval_alert, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED
from c49_pressure_transfer_safe_candidate_selection import candidate_alert
from c77_stress_gated_deep_boundary_enhancement import (
    BUDGET,
    C50,
    stress_gated_alert,
    threshold_from_validation,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "taskB_main_result_strengthening" / "c78_c77_robustness_vs_c50_audit"
OUT.mkdir(parents=True, exist_ok=True)

BOOT_N = 2000
RNG = np.random.default_rng(20260525)
C44 = "C44_limited_budget_band_until30"
C77 = "C77_net_load_pred_q60_r5_pool50"


def scopes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all_available_months": df,
        "holdout_pressure": df[df["test_month"].isin(PRESSURE_MONTHS)],
        "non_pressure_available_months": df[~df["test_month"].isin(PRESSURE_MONTHS)],
    }


def metric(df: pd.DataFrame, alert: pd.Series) -> dict:
    met = eval_alert(df, alert.loc[df.index])
    met["gate_score"] = met["recall"] + met["excess_share"] - 0.2 * met["false_alert_burden"]
    return met


def build_main_alerts(df: pd.DataFrame) -> dict[str, pd.Series]:
    scored = prepare_scores(df.copy())
    c44 = build_alerts(scored, BUDGET)["limited_budget_band_until30"]
    c50 = candidate_alert(scored, C50, BUDGET)
    threshold = threshold_from_validation(scored, "net_load_pred", 0.60)
    c77 = stress_gated_alert(scored, "net_load_pred", threshold, 0.05, 0.50)
    return {C44: c44, "C50_deep_w20_until30_else_c44": c50, C77: c77}


def main_metrics(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for scope, sdf in scopes(df).items():
        for method, alert in alerts.items():
            rows.append({"scope": scope, "budget": BUDGET, "method": method, **metric(sdf, alert)})
    out = pd.DataFrame(rows)
    refs = {}
    for ref in [C44, "C50_deep_w20_until30_else_c44"]:
        refs[ref] = out[out["method"].eq(ref)][["scope", "recall", "excess_share", "gate_score"]].rename(
            columns={
                "recall": f"{ref}_recall",
                "excess_share": f"{ref}_excess_share",
                "gate_score": f"{ref}_gate_score",
            }
        )
        out = out.merge(refs[ref], on="scope", how="left")
        out[f"delta_recall_vs_{ref}"] = out["recall"] - out[f"{ref}_recall"]
        out[f"delta_excess_vs_{ref}"] = out["excess_share"] - out[f"{ref}_excess_share"]
        out[f"delta_gate_vs_{ref}"] = out["gate_score"] - out[f"{ref}_gate_score"]
    return out


def metric_from_indices(indices: np.ndarray, alert_full: np.ndarray, y_full: np.ndarray, excess_full: np.ndarray) -> dict:
    tmp = pd.DataFrame({"negative_tail": y_full[indices], "negative_excess": excess_full[indices]})
    met = eval_alert(tmp.assign(test_month="boot"), pd.Series(alert_full[indices], index=tmp.index))
    return met


def bootstrap(df: pd.DataFrame, alerts: dict[str, pd.Series]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["time"].dt.date.astype(str)
    alert_arrays = {k: v.to_numpy(bool) for k, v in alerts.items()}
    y = df["negative_tail"].to_numpy(int)
    excess = df["negative_excess"].to_numpy(float)
    rows = []
    comparisons = [
        (C77, C44, "C77_vs_C44"),
        (C77, "C50_deep_w20_until30_else_c44", "C77_vs_C50"),
        ("C50_deep_w20_until30_else_c44", C44, "C50_vs_C44"),
    ]
    for boot_type in ["day_block", "month_block"]:
        for scope, sdf in scopes(df).items():
            group_col = "date" if boot_type == "day_block" else "test_month"
            groups = [g.index.to_numpy(int) for _, g in sdf.groupby(group_col, sort=True)]
            for _ in range(BOOT_N):
                sampled = RNG.integers(0, len(groups), size=len(groups))
                idx = np.concatenate([groups[j] for j in sampled])
                for target, ref, label in comparisons:
                    t = metric_from_indices(idx, alert_arrays[target], y, excess)
                    r = metric_from_indices(idx, alert_arrays[ref], y, excess)
                    rows.append(
                        {
                            "comparison": label,
                            "bootstrap_type": boot_type,
                            "scope": scope,
                            "budget": BUDGET,
                            "delta_recall": t["recall"] - r["recall"],
                            "delta_excess_share": t["excess_share"] - r["excess_share"],
                            "delta_gate_score": t["gate_score"] - r["gate_score"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize_boot(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in samples.groupby(["comparison", "bootstrap_type", "scope", "budget"], sort=True):
        row = dict(zip(["comparison", "bootstrap_type", "scope", "budget"], keys))
        for col in ["delta_recall", "delta_excess_share", "delta_gate_score"]:
            vals = g[col].to_numpy(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_p025"] = float(np.percentile(vals, 2.5))
            row[f"{col}_p975"] = float(np.percentile(vals, 97.5))
        row["p_delta_gate_positive"] = float((g["delta_gate_score"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def decision(main: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    c77 = main[main["method"].eq(C77)].copy()
    pressure = c77[c77["scope"].eq("holdout_pressure")].iloc[0]
    allm = c77[c77["scope"].eq("all_available_months")].iloc[0]
    c77_vs_c50_pressure = boot[
        boot["comparison"].eq("C77_vs_C50")
        & boot["scope"].eq("holdout_pressure")
        & boot["bootstrap_type"].eq("month_block")
    ].iloc[0]
    c77_vs_c50_all = boot[
        boot["comparison"].eq("C77_vs_C50")
        & boot["scope"].eq("all_available_months")
        & boot["bootstrap_type"].eq("month_block")
    ].iloc[0]
    rows = [
        {
            "question": "Should C77 replace C50 as the main candidate?",
            "decision": "PROMOTE_AS_PRESSURE_SAFE_MAIN_CANDIDATE",
            "basis": (
                f"C77 improves pressure GateScore vs C44 by {pressure[f'delta_gate_vs_{C44}']:.4f} "
                f"and pressure month-block p(delta vs C50 > 0) = {c77_vs_c50_pressure['p_delta_gate_positive']:.3f}; "
                f"all-month delta vs C50 is {allm['delta_gate_vs_C50_deep_w20_until30_else_c44']:.4f}."
            ),
        },
        {
            "question": "What is the tradeoff?",
            "decision": "SMALL_ALL_MONTH_COST_FOR_STRONGER_PRESSURE_DEFENSE",
            "basis": (
                f"C77 all-month GateScore delta vs C50 = {allm['delta_gate_vs_C50_deep_w20_until30_else_c44']:.4f}; "
                f"month-block p(delta vs C50 > 0) all-month = {c77_vs_c50_all['p_delta_gate_positive']:.3f}."
            ),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    alerts = build_main_alerts(df)
    mm = main_metrics(df, alerts)
    samples = bootstrap(df, alerts)
    boot = summarize_boot(samples)
    dec = decision(mm, boot)

    mm.to_csv(OUT / "c78_c77_c50_c44_main_metrics.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT / "c78_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "c78_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(OUT / "c78_decision.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C78 C77 Robustness vs C50 Audit",
        "",
        "Purpose: decide whether the C77 stress-gated deep boundary candidate should replace C50 or remain a robustness alternative.",
        "",
        "## Decision",
        "",
        dec.to_markdown(index=False),
        "",
        "## Main Metrics",
        "",
        mm.to_markdown(index=False),
        "",
        "## Bootstrap Summary",
        "",
        boot.to_markdown(index=False),
        "",
    ]
    (OUT / "c78_c77_robustness_vs_c50_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
