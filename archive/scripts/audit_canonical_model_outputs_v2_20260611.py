from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "canonical_model_audit_20260611_v2"
OUT.mkdir(exist_ok=True)

EVAL_MONTHS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
PRESSURE_MONTHS = {"2025-09", "2025-10"}
BUDGET = 0.30
GATE_FALSE_ALERT_WEIGHT = 0.2


@dataclass(frozen=True)
class ScoreModel:
    model_family: str
    model_name: str
    role: str
    source_code: str
    output_path: str
    score_col: str
    entry_type: str
    notes: str = ""


@dataclass(frozen=True)
class MetricsModel:
    model_family: str
    model_name: str
    role: str
    source_code: str
    metrics_path: str
    metrics_method: str
    entry_type: str
    notes: str = ""


def normalize_months(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "test_month" not in df.columns:
        raise ValueError("missing test_month")
    df["test_month"] = df["test_month"].astype(str)
    return df


def reference_counts() -> pd.DataFrame:
    ref_path = ROOT / "c20_distributional_tail_risk_scores" / "c20_distributional_predictions.csv"
    ref = normalize_months(pd.read_csv(ref_path, usecols=["test_month", "time", "negative_tail", "negative_excess"]))
    ref = ref[ref["test_month"].isin(EVAL_MONTHS)].copy()
    out = ref.groupby("test_month").size().rename("reference_n").reset_index()
    out.to_csv(OUT / "canonical_reference_month_counts.csv", index=False, encoding="utf-8-sig")
    return out


def rank_metrics(frame: pd.DataFrame, score_col: str) -> dict:
    alerts = pd.Series(False, index=frame.index)
    for _, g in frame.groupby("test_month"):
        k = int(np.ceil(BUDGET * len(g)))
        idx = g[score_col].astype(float).sort_values(ascending=False).index[:k]
        alerts.loc[idx] = True
    y = frame["negative_tail"].astype(int)
    ex = frame["negative_excess"].astype(float)
    events = int(y.sum())
    n_alerts = int(alerts.sum())
    hits = int((alerts & (y == 1)).sum())
    precision = hits / n_alerts if n_alerts else np.nan
    recall = hits / events if events else np.nan
    excess_share = float(ex[alerts].sum() / ex.sum()) if ex.sum() > 0 else np.nan
    false_alert_burden = float((alerts & (y == 0)).sum() / n_alerts) if n_alerts else np.nan
    gate_score = recall + excess_share - GATE_FALSE_ALERT_WEIGHT * false_alert_burden
    return {
        "n": int(len(frame)),
        "events": events,
        "alerts": n_alerts,
        "precision": precision,
        "recall": recall,
        "excess_share": excess_share,
        "false_alert_burden": false_alert_burden,
        "gate_score": gate_score,
    }


def completeness_status(df: pd.DataFrame, score_col: str, ref_counts: pd.DataFrame) -> tuple[str, str]:
    months = sorted(df["test_month"].dropna().astype(str).unique())
    missing = [m for m in EVAL_MONTHS if m not in months]
    count = df[df["test_month"].isin(EVAL_MONTHS)].groupby("test_month").size().rename("n").reset_index()
    merged = ref_counts.merge(count, on="test_month", how="left")
    mismatches = merged[merged["n"].fillna(-1).astype(int) != merged["reference_n"].astype(int)]
    if missing:
        return "rerun_or_exclude", f"missing months: {', '.join(missing)}"
    if not mismatches.empty:
        detail = "; ".join(
            f"{r.test_month}: n={int(r.n) if pd.notna(r.n) else 0}, ref={int(r.reference_n)}"
            for r in mismatches.itertuples()
        )
        return "rerun_or_exclude", f"interval count mismatch: {detail}"
    h = df[df["test_month"].isin(EVAL_MONTHS)]
    score_missing = h.loc[h[score_col].isna(), "test_month"].drop_duplicates().astype(str).tolist()
    if score_missing:
        return "conditional_ready", (
            f"score has early-window NA in months: {', '.join(score_missing)}; "
            "use only if NA handling is explicitly defined"
        )
    return "canonical_ready", "full canonical month, interval, label, and score support"


def score_models() -> list[ScoreModel]:
    c20 = "c20_distributional_tail_risk_scores/c20_distributional_predictions.csv"
    return [
        ScoreModel("Simple market-signal baseline", "Historical lower-tail ranking", "external baseline", "c20_distributional_tail_risk_scores.py", c20, "hist_tail65_rate_7d", "score_output"),
        ScoreModel("Market-mechanism baseline", "DA-congestion pressure ranking", "external baseline", "c20_distributional_tail_risk_scores.py", c20, "da_cong_negative_pressure", "score_output"),
        ScoreModel("Classical probabilistic ML", "HGB q10 downside-risk ranking", "external baseline", "c20_distributional_tail_risk_scores.py", c20, "c20_hgb_q10_risk", "score_output"),
        ScoreModel("Classical probabilistic ML", "Random-forest expected-shortfall ranking", "external baseline", "c20_distributional_tail_risk_scores.py", c20, "c20_rf_expected_shortfall", "score_output"),
        ScoreModel("Classical probabilistic ML", "LightGBM q10 distance ranking", "external baseline", "model_family_complementarity_audit.py", "model_family_complementarity_audit/merged_family_scores.csv", "LightGBM_q10_distance_two_stage", "score_output"),
        ScoreModel("VMD recurrent deep model", "LSTM-VMD risk ranking, short window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "lstm_s48_h64_w4_vmd", "score_output"),
        ScoreModel("VMD recurrent deep model", "LSTM-VMD risk ranking, long window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "lstm_s96_h96_w6_vmd", "score_output"),
        ScoreModel("VMD recurrent deep model", "GRU-VMD risk ranking, short window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "gru_s48_h64_w6_vmd", "score_output"),
        ScoreModel("VMD recurrent deep model", "GRU-VMD risk ranking, long window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "gru_s96_h96_w8_vmd", "score_output"),
        ScoreModel("VMD temporal-convolution deep model", "TCN-VMD risk ranking, short window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "tcn_s48_h64_w6_vmd", "score_output"),
        ScoreModel("VMD temporal-convolution deep model", "TCN-VMD risk ranking, long window", "external deep baseline", "run_slts_canonical_experiment.py", c20, "tcn_s96_h96_w8_vmd", "score_output"),
        ScoreModel("Standalone LSTM quantile model", "LSTM quantile probability ranking", "external deep baseline", "qia_tail_lstm_quantile_module.py", "qia_tail_lstm_quantile_module/lstm_quantile_predictions.csv", "calibrated_probability", "score_output"),
        ScoreModel("Standalone TCN quantile model", "TCN quantile probability ranking", "external deep baseline", "qia_tail_tcn_quantile_module.py", "qia_tail_tcn_quantile_module/tcn_quantile_predictions.csv", "calibrated_probability", "score_output"),
        ScoreModel("Standalone Transformer quantile model", "Transformer quantile probability ranking", "external deep baseline", "qia_tail_transformer_quantile_module.py", "qia_tail_transformer_quantile_module/transformer_quantile_predictions.csv", "calibrated_probability", "score_output"),
        ScoreModel("Selective state-space sequence model", "Selective SSM-isotonic blend ranking", "external sequence baseline", "c59_local_selective_ssm_tail_baseline.py", "c59_local_selective_ssm_tail_baseline/c59_local_selective_ssm_predictions.csv", "c59_blend_c59_selssm_tail_prob_iso_base_w0p2", "score_output", "full-window score is the base-blended calibrated SSM column; raw SSM score has early-window NA"),
        ScoreModel("Dilated TCN attention model", "Dilated TCN attention-isotonic blend ranking", "external deep baseline", "c60_dilated_tcn_attention_tail_baseline.py", "c60_dilated_tcn_attention_tail_baseline/c60_dilated_tcn_attention_predictions.csv", "c60_blend_c60_tcn_tail_prob_iso_base_w0p2", "score_output", "full-window score is the base-blended calibrated TCN-attention column; raw model has early-window NA"),
        ScoreModel("Pressure-aware TCN attention model", "Pressure-aware TCN attention-isotonic blend ranking", "external deep baseline", "c61_pressure_aware_tcn_attention_tail_baseline.py", "c61_pressure_aware_tcn_attention_tail_baseline/c61_pressure_aware_tcn_attention_predictions.csv", "c61_blend_c61_tcn_tail_prob_iso_base_w0p2", "score_output", "full-window score is the base-blended calibrated pressure-aware TCN column; raw model has early-window NA"),
        ScoreModel("Deep distributional post-processing", "Quantile-MLP post-processing", "external probabilistic baseline", "c31_conformalized_deep_ensemble.py", "c31_conformalized_deep_ensemble/c31_conformalized_deep_ensemble_predictions.csv", "c31_blend_c31_tail_prob_iso_base_w0p2", "score_output"),
        ScoreModel("Conditional diffusion scenario model", "Conditional diffusion q20-risk ranking", "external diffusion baseline", "c33_conditional_diffusion_residual_baseline.py", "c33_conditional_diffusion_residual_baseline/c33_conditional_diffusion_predictions.csv", "c33_diff_q20_risk", "score_output"),
        ScoreModel("Conditional diffusion scenario model", "Conditional diffusion-isotonic blend ranking", "external diffusion baseline", "c33_conditional_diffusion_residual_baseline.py", "c33_conditional_diffusion_residual_baseline/c33_conditional_diffusion_predictions.csv", "c33_blend_c33_diff_q20_risk_base_w0p2", "score_output"),
        ScoreModel("Tail-aware residual diffusion model", "Tail-aware residual diffusion probability ranking", "external diffusion baseline", "tail_aware_residual_diffusion_slts_remote.py", "tail_aware_residual_diffusion_outputs/tail_aware_residual_diffusion_predictions.csv", "TailAwareDiffusion_p_negative", "score_output"),
        ScoreModel("Diffusion distribution module", "Diffusion-QIA tail-probability ranking", "external diffusion baseline", "qia_tail_diffusion_distribution_module.py", "qia_tail_diffusion_distribution_module/diffusion_qia_predictions.csv", "diffusion_tail_probability", "score_output"),
        ScoreModel("Signal stacking diagnostic", "Tree-deep-diffusion logistic stacking", "supplementary diagnostic", "unified_signal_stacking_experiment.py", "unified_signal_stacking_experiment/unified_stacking_predictions.csv", "Stack_TreeDeepDiff_plain", "score_output", "old stacking output starts in December and cannot support pressure-month claims"),
    ]


def metrics_models() -> list[MetricsModel]:
    return [
        MetricsModel("Calibrated distributional ML", "HGB-isotonic tail ranking", "operational reference", "c54_top30_final_robustness.py", "c54_top30_final_robustness/c54_top30_main_metrics.csv", "C20b_base", "metric_table", "paper-ready operational reference; do not replace with raw c20_iso_tail_prob"),
        MetricsModel("Boundary-aware ranking", "DA-congestion full ranking", "strong mechanism-aware baseline", "c52_manuscript_result_package.py", "paper_ready_results/baseline_table.csv", "C22_full_rank", "metric_table"),
        MetricsModel("Fixed-budget screening component", "Core-preserved fixed-budget screening", "internal component", "c44_validation_safe_limited_budget_rule.py", "c54_top30_final_robustness/c54_top30_main_metrics.csv", "C44_limited_budget_band_until30", "metric_table"),
        MetricsModel("Deep boundary component", "Quantile-MLP boundary reranking", "internal component", "c50_conservative_parsimony_selection_and_bootstrap.py", "c54_top30_final_robustness/c54_top30_main_metrics.csv", "c22_deep_w20_until30_else_c44", "metric_table"),
        MetricsModel("Stress-gated boundary component", "Net-load-gated Quantile-MLP boundary replacement", "retained method", "c77/c78/c79/c81 chain", "taskB_main_result_strengthening/c79_c77_paper_ready_integration/c79_main_results_with_c77.csv", "C77 pressure-safe stress-gated candidate", "metric_table", "selection corrected by C81 validation-only parsimony rule"),
    ]


def audit_score_models(models: list[ScoreModel], ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_rows: list[dict] = []
    met_rows: list[dict] = []
    for m in models:
        path = ROOT / m.output_path
        row = m.__dict__.copy()
        if not path.exists():
            row.update(status="rerun_or_exclude", status_detail="output file missing")
            inv_rows.append(row)
            continue
        try:
            df = normalize_months(pd.read_csv(path))
        except Exception as exc:
            row.update(status="rerun_or_exclude", status_detail=f"cannot read output: {exc}")
            inv_rows.append(row)
            continue
        if m.score_col not in df.columns:
            row.update(status="rerun_or_exclude", status_detail=f"score column missing: {m.score_col}")
            inv_rows.append(row)
            continue
        if "negative_tail" not in df.columns or "negative_excess" not in df.columns:
            row.update(status="rerun_or_exclude", status_detail="missing negative_tail or negative_excess")
            inv_rows.append(row)
            continue
        status, detail = completeness_status(df, m.score_col, ref)
        row.update(status=status, status_detail=detail, available_months=", ".join(sorted(df["test_month"].astype(str).unique())))
        inv_rows.append(row)
        usable = df[df["test_month"].isin(EVAL_MONTHS)].copy()
        if status == "rerun_or_exclude":
            continue
        for scope, g in [
            ("all_available_months", usable),
            ("holdout_pressure", usable[usable["test_month"].isin(PRESSURE_MONTHS)]),
            ("non_pressure_available_months", usable[~usable["test_month"].isin(PRESSURE_MONTHS)]),
        ]:
            if g.empty:
                continue
            met_rows.append({**m.__dict__, "status": status, "scope": scope, "budget": BUDGET, **rank_metrics(g, m.score_col)})
    return pd.DataFrame(inv_rows), pd.DataFrame(met_rows)


def audit_metric_tables(models: list[MetricsModel]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_rows: list[dict] = []
    met_rows: list[dict] = []
    for m in models:
        path = ROOT / m.metrics_path
        row = m.__dict__.copy()
        if not path.exists():
            row.update(status="rerun_or_exclude", status_detail="metrics table missing")
            inv_rows.append(row)
            continue
        df = pd.read_csv(path)
        if "method" in df.columns:
            h = df[df["method"].astype(str).eq(m.metrics_method)].copy()
        elif "display_name" in df.columns:
            h = df[df["display_name"].astype(str).eq(m.metrics_method)].copy()
        else:
            h = pd.DataFrame()
        if h.empty:
            row.update(status="rerun_or_exclude", status_detail=f"metrics method not found: {m.metrics_method}")
            inv_rows.append(row)
            continue
        row.update(status="metric_ready", status_detail="paper-ready metric table found")
        inv_rows.append(row)
        for _, r in h.iterrows():
            if "scope" not in r and {"all_recall", "pressure_recall"}.issubset(set(h.columns)):
                for scope, prefix in [
                    ("all_available_months", "all"),
                    ("holdout_pressure", "pressure"),
                ]:
                    met_rows.append(
                        {
                            **m.__dict__,
                            "status": "metric_ready",
                            "scope": scope,
                            "budget": BUDGET,
                            "n": np.nan,
                            "events": np.nan,
                            "alerts": np.nan,
                            "precision": np.nan,
                            "recall": r.get(f"{prefix}_recall", np.nan),
                            "excess_share": r.get(f"{prefix}_excess_share", np.nan),
                            "false_alert_burden": np.nan,
                            "gate_score": r.get(f"{prefix}_gate_score", np.nan),
                        }
                    )
                continue
            met_rows.append(
                {
                    **m.__dict__,
                    "status": "metric_ready",
                    "scope": r.get("scope"),
                    "budget": r.get("budget", BUDGET),
                    "n": r.get("n", np.nan),
                    "events": r.get("events", np.nan),
                    "alerts": r.get("alerts", np.nan),
                    "precision": r.get("precision", np.nan),
                    "recall": r.get("recall", r.get("all_recall", np.nan)),
                    "excess_share": r.get("excess_share", r.get("all_excess_share", np.nan)),
                    "false_alert_burden": r.get("false_alert_burden", np.nan),
                    "gate_score": r.get("gate_score", r.get("all_gate_score", np.nan)),
                }
            )
    return pd.DataFrame(inv_rows), pd.DataFrame(met_rows)


def build_wide_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in metrics.groupby(["model_family", "model_name", "role", "entry_type"], dropna=False):
        family, name, role, entry_type = keys
        row = {"model_family": family, "model_name": name, "role": role, "entry_type": entry_type}
        for scope, prefix in [
            ("all_available_months", "all"),
            ("holdout_pressure", "pressure"),
            ("non_pressure_available_months", "non_pressure"),
        ]:
            h = g[g["scope"].astype(str).eq(scope)]
            if not h.empty:
                r = h.iloc[0]
                row[f"{prefix}_recall"] = r["recall"]
                row[f"{prefix}_excess_share"] = r["excess_share"]
                row[f"{prefix}_gate_score"] = r["gate_score"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        role_order = {
            "retained method": 0,
            "internal component": 1,
            "operational reference": 2,
            "strong mechanism-aware baseline": 3,
            "external probabilistic baseline": 4,
            "external diffusion baseline": 5,
            "external deep baseline": 6,
            "external sequence baseline": 7,
            "external baseline": 8,
            "supplementary diagnostic": 9,
        }
        out["_role_order"] = out["role"].map(role_order).fillna(99)
        out = out.sort_values(["_role_order", "model_family", "model_name"]).drop(columns=["_role_order"])
    return out


def audit_nyiso() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = ROOT / "taskD_nyiso_method_transfer_20260610" / "nyiso_transfer_dataset.csv"
    metrics = ROOT / "taskD_nyiso_method_transfer_20260610" / "nyiso_method_transfer_metrics.csv"
    info_rows = []
    if dataset.exists():
        df = pd.read_csv(dataset, usecols=["timestamp", "zone", "lower_tail_event", "negative_excess", "base_score", "congestion_score", "sequence_score", "boundary_score", "load_forecast"])
        info_rows.append(
            {
                "item": "nyiso_transfer_dataset",
                "exists": True,
                "n": len(df),
                "zones": df["zone"].nunique(),
                "time_min": df["timestamp"].min(),
                "time_max": df["timestamp"].max(),
                "events": int(df["lower_tail_event"].sum()),
                "negative_excess_sum": float(df["negative_excess"].sum()),
                "available_scores": "base_score; congestion_score; sequence_score; boundary_score; load_forecast",
            }
        )
    else:
        info_rows.append({"item": "nyiso_transfer_dataset", "exists": False})
    if metrics.exists():
        m = pd.read_csv(metrics)
        method_rows = []
        for method, g in m[m["scope"].eq("all_confirmation_zone_months")].groupby("method"):
            r = g.iloc[0]
            method_rows.append(
                {
                    "model_family": "NYISO local-transfer experiment",
                    "model_name": method,
                    "scope": r["scope"],
                    "n": r["n"],
                    "events": r["events"],
                    "alerts": r["alerts"],
                    "recall": r["recall"],
                    "excess_share": r["excess_share"],
                    "false_alert_burden": r["false_alert_burden"],
                    "gate_score": r["gate_score"],
                }
            )
        return pd.DataFrame(info_rows), pd.DataFrame(method_rows)
    return pd.DataFrame(info_rows), pd.DataFrame()


def main() -> None:
    ref = reference_counts()
    score_inv, score_met = audit_score_models(score_models(), ref)
    table_inv, table_met = audit_metric_tables(metrics_models())
    inv = pd.concat([score_inv, table_inv], ignore_index=True)
    met = pd.concat([score_met, table_met], ignore_index=True)
    wide = build_wide_table(met)
    nyiso_info, nyiso_methods = audit_nyiso()

    inv.to_csv(OUT / "model_inventory_v2.csv", index=False, encoding="utf-8-sig")
    met.to_csv(OUT / "model_metrics_long_v2.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT / "section5_model_result_candidates_v2.csv", index=False, encoding="utf-8-sig")
    nyiso_info.to_csv(OUT / "nyiso_transfer_dataset_audit_v2.csv", index=False, encoding="utf-8-sig")
    nyiso_methods.to_csv(OUT / "nyiso_transfer_method_metrics_v2.csv", index=False, encoding="utf-8-sig")

    ready = inv[inv["status"].isin(["canonical_ready", "metric_ready"])].copy()
    conditional = inv[inv["status"].eq("conditional_ready")].copy()
    rerun = inv[inv["status"].eq("rerun_or_exclude")].copy()
    with open(OUT / "model_audit_summary_v2.md", "w", encoding="utf-8") as f:
        f.write("# Model-family audit v2, 2026-06-11\n\n")
        f.write("This audit is code-grounded. It separates true score outputs, paper-ready metric tables, internal components, external baselines, and retained variants.\n\n")
        f.write("## Ready for main-table consideration\n\n")
        f.write(ready[["model_family", "model_name", "role", "entry_type", "source_code", "status_detail"]].to_markdown(index=False) if not ready.empty else "None.")
        f.write("\n\n## Conditional entries\n\n")
        f.write(conditional[["model_family", "model_name", "role", "entry_type", "source_code", "status_detail", "notes"]].to_markdown(index=False) if not conditional.empty else "None.")
        f.write("\n\n## Rerun or exclude\n\n")
        f.write(rerun[["model_family", "model_name", "role", "entry_type", "source_code", "status_detail"]].to_markdown(index=False) if not rerun.empty else "None.")
        f.write("\n\n## NYISO transfer status\n\n")
        if not nyiso_info.empty:
            f.write(nyiso_info.to_markdown(index=False))
        if not nyiso_methods.empty:
            f.write("\n\n")
            f.write(nyiso_methods.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Interpretation rules\n\n")
        f.write("- `metric_ready` entries are valid paper-ready metric summaries but require their generating scripts and selection rules to be described in the manuscript or appendix.\n")
        f.write("- `canonical_ready` entries have full interval-level score outputs under the Zhejiang canonical window.\n")
        f.write("- `conditional_ready` entries have complete rows but need explicit NA/window handling before they can be defended.\n")
        f.write("- The final Section 5 table must not mix exploratory stacking or partial-window results with full-window canonical results.\n")
        f.write("- The strongest paper-specific result is C77 only within the pressure-slice/top30 claim boundary; C50 remains the all-month comparator.\n")

    print("ready", len(ready), "conditional", len(conditional), "rerun", len(rerun))
    print("outputs", OUT)
    print(wide[["model_family", "model_name", "role", "all_recall", "all_excess_share", "all_gate_score", "pressure_recall", "pressure_excess_share", "pressure_gate_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
