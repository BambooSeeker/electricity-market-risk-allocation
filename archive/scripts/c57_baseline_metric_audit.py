from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c57_baseline_metric_audit"
OUT.mkdir(exist_ok=True)


def baseline_audit() -> pd.DataFrame:
    rows = [
        {
            "method": "C20b_base",
            "paper_role": "Base distributional tail-risk ranking",
            "implementation": "HGB quantile/mean models, RF scenario distribution, HGB tail classifier with isotonic calibration",
            "training_protocol": "rolling month folds; each evaluated month trains only on pre-test history; previous month used for calibration where available",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low",
            "scientific_status": "solid baseline",
            "main_text_claim": "allowed_as_base_risk_ranking",
            "weakness": "not a deep model; mainly establishes calibrated distributional risk signal",
            "required_action": "keep",
        },
        {
            "method": "C22_full_rank",
            "paper_role": "Full boundary-aware reranking",
            "implementation": "base distributional score plus ranked day-ahead congestion-boundary bonus",
            "training_protocol": "fixed alpha variants and diagnostic transfer checks; later C44 provides validation-safe limited-budget version",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "medium_if_written_as_final_method; low_if_presented_as_mechanism_baseline",
            "scientific_status": "useful mechanism-aware baseline",
            "main_text_claim": "allowed_as_mechanism_baseline_not_final_claim",
            "weakness": "full reranking is less protected than C44; should not be the final method",
            "required_action": "keep_with_boundary_language",
        },
        {
            "method": "c31_blend_c31_tail_prob_iso_base_w0p2",
            "paper_role": "Post-processed probabilistic baseline",
            "implementation": "deep quantile MLP ensemble with BCE tail head, conformalized quantiles, isotonic probability calibration, blended into base ranking",
            "training_protocol": "rolling month folds; train before test month; previous month calibration; ensemble seeds",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low",
            "scientific_status": "credible modern probabilistic baseline",
            "main_text_claim": "allowed_as_postprocessing_or_conformal_deep_baseline",
            "weakness": "MLP architecture is not itself state-of-the-art in 2026; value is calibration/post-processing rather than model novelty",
            "required_action": "keep_but_do_not_call_frontier_architecture",
        },
        {
            "method": "c33_blend_c33_diff_q20_risk_base_w0p2",
            "paper_role": "Conditional generative risk baseline",
            "implementation": "conditional diffusion MLP over scalar spread distribution; 40 denoising steps; 96 scenarios per timestamp; blended q20 risk",
            "training_protocol": "rolling month folds; train only before test month; fixed sampling protocol",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low",
            "scientific_status": "acceptable as generative baseline but not a strong frontier diffusion claim",
            "main_text_claim": "allowed_as_representative_generative_baseline",
            "weakness": "scalar conditional diffusion is simpler than modern time-series diffusion; weak pressure results; should not carry novelty",
            "required_action": "keep_as_baseline_or_appendix; consider stronger generative baseline only if extra evidence is needed",
        },
        {
            "method": "c32_blend_c32_ssm_tail_prob_iso_base_w0p2",
            "paper_role": "State-space-like probabilistic baseline",
            "implementation": "custom gated diagonal state recurrence with attention pooling; quantile and tail-prob heads; single ensemble seed",
            "training_protocol": "rolling month folds; train before test month; previous month calibration where possible",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low",
            "scientific_status": "acceptable as SSM-like diagnostic baseline, weak as frontier SSM evidence",
            "main_text_claim": "allowed_only_as_state_space_like_baseline",
            "weakness": "not a real Mamba/S4 implementation; single seed; weak pressure results",
            "required_action": "keep_but_label_SSM_like; do_not_claim_Mamba_or_frontier_SSM",
        },
        {
            "method": "C44_limited_budget_band_until30",
            "paper_role": "Robust conservative anchor",
            "implementation": "validation-frozen limited-budget boundary rule; preserves base outside the limited operating band",
            "training_protocol": "selected through validation-safe rule; pressure months reported as holdout evidence",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low_after_C44_freeze",
            "scientific_status": "robust anchor",
            "main_text_claim": "allowed_as_conservative_anchor",
            "weakness": "pressure bootstrap crosses zero; leave-one-pressure-month-out depends on 2025-10",
            "required_action": "keep_with_uncertainty_language",
        },
        {
            "method": "c22_deep_w20_until30_else_c44",
            "paper_role": "Deep-sequence top30 enhanced rule",
            "implementation": "top30 budget uses deep-sequence risk signal; outside top30 reverts to C44",
            "training_protocol": "selected after conservative parsimony and robustness checks; numerically same as c22_deep_w20 at top30",
            "fixed_budget_eval": "yes",
            "pressure_label_leakage_risk": "low_if_top30_claim_is_prespecified",
            "scientific_status": "main-budget enhanced candidate",
            "main_text_claim": "allowed_as_top30_enhanced_candidate",
            "weakness": "not multi-budget dominant; pressure bootstrap only weakly positive; identical to deep signal under top30",
            "required_action": "keep_as_top30_candidate; explicitly_avoid_duplicate_counting",
        },
    ]
    return pd.DataFrame(rows)


def metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "symbol": "A_t",
            "name": "alert indicator",
            "definition": "A_t = 1 if timestamp t is selected into the fixed-budget alert set within its test month, otherwise 0.",
            "role": "protocol",
            "main_or_aux": "main",
        },
        {
            "symbol": "Y_t",
            "name": "lower-tail event indicator",
            "definition": "Y_t = 1 if spread_rt_minus_da_t <= tau_negative_m for the corresponding test month m, otherwise 0.",
            "role": "risk event",
            "main_or_aux": "main",
        },
        {
            "symbol": "L_t^-",
            "name": "negative excess",
            "definition": "L_t^- = max(tau_negative_m - spread_rt_minus_da_t, 0).",
            "role": "loss-severity proxy",
            "main_or_aux": "main",
        },
        {
            "symbol": "Recall",
            "name": "event coverage",
            "definition": "sum_t A_t Y_t / sum_t Y_t.",
            "role": "event coverage under fixed budget",
            "main_or_aux": "main",
        },
        {
            "symbol": "ExcessShare",
            "name": "excess-loss coverage",
            "definition": "sum_t A_t L_t^- / sum_t L_t^-.",
            "role": "severity coverage under fixed budget",
            "main_or_aux": "main",
        },
        {
            "symbol": "Precision",
            "name": "alert precision",
            "definition": "sum_t A_t Y_t / sum_t A_t.",
            "role": "alert purity",
            "main_or_aux": "auxiliary",
        },
        {
            "symbol": "FalseAlertBurden",
            "name": "false-alert burden",
            "definition": "1 - Precision.",
            "role": "cost proxy of limited alert resource",
            "main_or_aux": "auxiliary",
        },
        {
            "symbol": "GateScore",
            "name": "summary gate score",
            "definition": "Recall + ExcessShare - 0.2 * FalseAlertBurden.",
            "role": "compact reporting score, not the sole objective",
            "main_or_aux": "auxiliary_summary",
        },
    ]
    return pd.DataFrame(rows)


def reviewer_audit() -> pd.DataFrame:
    rows = [
        {
            "reviewer_question": "Is top30 chosen after seeing the result?",
            "risk_level": "high",
            "current_answer": "Top30 is now the main operational alert budget, but it must be justified as prespecified and other budgets must remain in appendix.",
            "next_action": "write top30 as operational budget; keep budget sensitivity visible",
        },
        {
            "reviewer_question": "Are modern baselines strong enough for 2026?",
            "risk_level": "medium_high",
            "current_answer": "C31 is credible; C33 and C32 are representative but simplified. They defend technical coverage but cannot support a frontier-model claim.",
            "next_action": "label C32 as SSM-like and C33 as representative generative; consider one stronger baseline only if final score remains below threshold",
        },
        {
            "reviewer_question": "Does the paper overclaim significance on pressure months?",
            "risk_level": "high",
            "current_answer": "No if C56 boundary is followed. Pressure-month bootstrap crosses zero, so claims must be directional and bounded.",
            "next_action": "avoid significant/dominant wording for pressure months",
        },
        {
            "reviewer_question": "Is the method just model stacking?",
            "risk_level": "medium",
            "current_answer": "The main story is fixed-budget risk coverage with validation-safe boundary protection, not a new architecture. C50 is a top30 enhanced rule, not a grand model invention.",
            "next_action": "describe rules/protocol instead of packaging a named framework",
        },
        {
            "reviewer_question": "Is price_day_ahead_cong an unfair feature?",
            "risk_level": "medium",
            "current_answer": "It is treated as an available market/mechanism feature, not as an explicit outage label. The paper should not say it directly encodes maintenance.",
            "next_action": "write feature availability and mechanism boundary carefully",
        },
    ]
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    view = frame[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    audit = baseline_audit()
    metrics = metric_definitions()
    review = reviewer_audit()

    audit.to_csv(OUT / "c57_baseline_implementation_audit.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "c57_metric_definition_freeze.csv", index=False, encoding="utf-8-sig")
    review.to_csv(OUT / "c57_ai_reviewer_reverse_audit.csv", index=False, encoding="utf-8-sig")

    report = [
        "# C57 Baseline Implementation Audit and Metric-Definition Freeze",
        "",
        "Purpose: audit whether the current main-text baselines are fair, scientific, and not over-claimed; freeze the metric definitions before manuscript drafting.",
        "",
        "## Baseline Implementation Audit",
        "",
        md_table(
            audit,
            [
                "method",
                "paper_role",
                "scientific_status",
                "main_text_claim",
                "weakness",
                "required_action",
            ],
            20,
        ),
        "",
        "## Frozen Metric Definitions",
        "",
        md_table(metrics, ["symbol", "name", "definition", "role", "main_or_aux"], 20),
        "",
        "## AI Reviewer Reverse Audit",
        "",
        md_table(review, ["reviewer_question", "risk_level", "current_answer", "next_action"], 20),
        "",
        "## Interpretation",
        "",
        "- C31 is the most defensible modern probabilistic baseline because it includes ensemble quantiles, conformalization, and isotonic calibration.",
        "- C32 and C33 are acceptable as representative SSM-like/generative baselines, but they are not strong enough to support frontier-model claims.",
        "- GateScore is frozen as an auxiliary compact score; Recall and ExcessShare remain the substantive main metrics.",
        "- The next experimental decision is whether to strengthen one modern baseline or proceed with the current bounded claim.",
    ]
    (OUT / "c57_baseline_metric_audit_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
