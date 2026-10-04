from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_result_tables"
OUT.mkdir(parents=True, exist_ok=True)


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def fmt_num(x, digits=3):
    return f"{x:.{digits}f}"


def load_csv(path):
    return pd.read_csv(ROOT / path)


def make_stage_ab_table():
    df = load_csv("rolling_tail_risk_validation/rolling_summary.csv")
    rows = []
    for _, r in df.sort_values(["stage", "model"]).iterrows():
        rows.append(
            {
                "stage": r["stage"],
                "model": r["model"],
                "AUC": fmt_num(r["auc_mean"]),
                "AP": fmt_num(r["average_precision_mean"]),
                "Brier": fmt_num(r["brier_mean"]),
                "Top5 Precision": fmt_pct(r["top5_precision_mean"]),
                "Top5 Lift": fmt_num(r["top5_lift_mean"]),
                "Top20 Recall": fmt_pct(r["top20_recall_mean"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table1_stage_ab_overall_tail.csv", index=False, encoding="utf-8-sig")
    return table


def make_task_table():
    df = load_csv("unified_tail_task_comparison/unified_summary.csv")
    key = df[
        (df["stage"] == "StageB")
        & (df["model"] == "RF")
        & (df["framework"] == "single_task")
    ].copy()
    task_order = {"absolute_tail": 0, "positive_tail": 1, "negative_tail": 2}
    key["order"] = key["target"].map(task_order)
    key = key.sort_values("order")
    rows = []
    for _, r in key.iterrows():
        rows.append(
            {
                "task": r["target"],
                "AUC": fmt_num(r["auc_mean"]),
                "AP": fmt_num(r["average_precision_mean"]),
                "Top5 Precision": fmt_pct(r["top5_precision_mean"]),
                "Top5 Lift": fmt_num(r["top5_lift_mean"]),
                "Top10 Precision": fmt_pct(r["top10_precision_mean"]),
                "Top10 Lift": fmt_num(r["top10_lift_mean"]),
                "Top20 Recall": fmt_pct(r["top20_recall_mean"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table2_stageb_rf_asymmetric_tasks.csv", index=False, encoding="utf-8-sig")
    return table


def make_raw_calibrated_table():
    df = load_csv("calibrated_multihead_risk_model/multihead_summary.csv")
    key = df[(df["stage"] == "StageB") & (df["task"] == "negative_tail")].copy()
    key = key.sort_values(["head_kind", "score_type"])
    rows = []
    for _, r in key.iterrows():
        rows.append(
            {
                "head": r["head_kind"],
                "score": r["score_type"],
                "AUC": fmt_num(r["auc_mean"]),
                "AP": fmt_num(r["average_precision_mean"]),
                "Brier": fmt_num(r["brier_mean"]),
                "Log Loss": fmt_num(r["log_loss_mean"]),
                "Top5 Lift": fmt_num(r["top5_lift_mean"]),
                "Top20 Recall": fmt_pct(r["top20_recall_mean"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table3_negative_tail_raw_vs_calibrated.csv", index=False, encoding="utf-8-sig")
    return table


def make_reliability_table():
    df = load_csv("metric_reliability_audit/key_stageb_single_task_findings.csv")
    table = df.copy()
    table.to_csv(OUT / "table4_metric_reliability_flags.csv", index=False, encoding="utf-8-sig")
    return table


def markdown_table(df):
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(rows)


def write_report(t1, t2, t3, t4):
    lines = []
    lines.append("# Paper Result Tables")
    lines.append("")
    lines.append("## Table 1. Stage A vs Stage B Overall Tail-Risk Forecasting")
    lines.append("")
    lines.append(markdown_table(t1))
    lines.append("")
    lines.append("Interpretation: Stage B uses post-day-ahead-clearing but pre-real-time information. Improvements over Stage A should be interpreted as incremental value of day-ahead clearing information, not as use of ex-post real-time information.")
    lines.append("")
    lines.append("## Table 2. Stage B / RF Asymmetric Tail Tasks")
    lines.append("")
    lines.append(markdown_table(t2))
    lines.append("")
    lines.append("Interpretation: negative_tail has the strongest AUC and top-k recall/lift profile, supporting asymmetric tail-risk modeling rather than a single symmetric spread-tail task.")
    lines.append("")
    lines.append("## Table 3. Stage B Negative Tail: Raw Scores vs Calibrated Probabilities")
    lines.append("")
    lines.append(markdown_table(t3))
    lines.append("")
    lines.append("Interpretation: raw scores are mainly used for ranking and alerting, while calibrated scores are mainly used for probability quality. A drop in AUC/AP after calibration is not necessarily a failure if Brier/log loss improve.")
    lines.append("")
    lines.append("## Table 4. Metric Reliability Flags")
    lines.append("")
    lines.append(markdown_table(t4))
    lines.append("")
    lines.append("Interpretation: month-level folds with few tail events are diagnostic only. High base-rate folds are high-volatility regimes, not rare-event alert regimes.")
    lines.append("")
    lines.append("## Writing Guardrails")
    lines.append("")
    lines.append("- Do not call AUC accuracy.")
    lines.append("- Do not call lift a profit multiplier.")
    lines.append("- Do not use calibrated probability metrics to replace ranking metrics.")
    lines.append("- Do not present low-event monthly folds as strong standalone evidence.")
    (OUT / "paper_result_tables_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    t1 = make_stage_ab_table()
    t2 = make_task_table()
    t3 = make_raw_calibrated_table()
    t4 = make_reliability_table()
    write_report(t1, t2, t3, t4)
    print(OUT / "paper_result_tables_report.md")


if __name__ == "__main__":
    main()
