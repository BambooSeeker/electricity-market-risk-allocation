from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "taskB_main_result_strengthening" / "c79_c77_paper_ready_integration"
OUT.mkdir(parents=True, exist_ok=True)

C72_MAIN = ROOT / "c72_pre_manuscript_result_freeze_package" / "c72_frozen_main_result_table.csv"
C78_MAIN = ROOT / "taskB_main_result_strengthening" / "c78_c77_robustness_vs_c50_audit" / "c78_c77_c50_c44_main_metrics.csv"
C78_BOOT = ROOT / "taskB_main_result_strengthening" / "c78_c77_robustness_vs_c50_audit" / "c78_bootstrap_summary.csv"
C75_CURVE = ROOT / "taskB_main_result_strengthening" / "c75_hard_review_budget_stress_audit" / "c75_budget_curve_metrics.csv"
C77_SCREEN = ROOT / "taskB_main_result_strengthening" / "c77_stress_gated_deep_boundary_enhancement" / "c77_candidate_screen.csv"
C76_BOUNDARY = ROOT / "taskB_main_result_strengthening" / "c76_deployment_boundary_audit" / "c76_deployment_boundary_decision.csv"


def build_main_results() -> pd.DataFrame:
    old = pd.read_csv(C72_MAIN)
    c78 = pd.read_csv(C78_MAIN)
    base = old[old["display_name"].eq("Base distributional tail-risk ranking")].copy()
    base["paper_role"] = "operational_baseline"

    mapping = {
        "C44_limited_budget_band_until30": ("C44 robust conservative anchor", "general_budget_anchor"),
        "C50_deep_w20_until30_else_c44": ("C50 all-month stronger comparator", "all_month_comparator"),
        "C77_net_load_pred_q60_r5_pool50": ("C77 pressure-safe stress-gated candidate", "pressure_safe_main_candidate"),
    }
    rows = []
    for _, r in c78.iterrows():
        if r["method"] not in mapping:
            continue
        display, role = mapping[r["method"]]
        rows.append(
            {
                "scope": r["scope"],
                "display_name": display,
                "paper_role": role,
                "events": int(r["events"]),
                "alerts": int(r["alerts"]),
                "precision": r["precision"],
                "recall": r["recall"],
                "excess_share": r["excess_share"],
                "false_alert_burden": r["false_alert_burden"],
                "gate_score": r["gate_score"],
                "delta_gate_vs_base": pd.NA,
                "delta_gate_vs_c44": r["delta_gate_vs_C44_limited_budget_band_until30"],
                "delta_gate_vs_c50": r["delta_gate_vs_C50_deep_w20_until30_else_c44"],
            }
        )
    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False)
    # Fill deltas against the base for C44/C50/C77 using scope-level base gate.
    base_gate = out[out["display_name"].eq("Base distributional tail-risk ranking")][["scope", "gate_score"]].rename(
        columns={"gate_score": "base_gate_score"}
    )
    out = out.merge(base_gate, on="scope", how="left")
    out["delta_gate_vs_base"] = out["gate_score"] - out["base_gate_score"]
    out.loc[out["display_name"].eq("Base distributional tail-risk ranking"), "delta_gate_vs_base"] = 0.0
    out.loc[out["display_name"].eq("Base distributional tail-risk ranking"), "delta_gate_vs_c44"] = (
        out.loc[out["display_name"].eq("Base distributional tail-risk ranking"), "delta_gate_vs_c44"].astype(float)
    )
    out["delta_gate_vs_c50"] = out["delta_gate_vs_c50"].fillna(pd.NA)
    out = out.drop(columns=["base_gate_score"])
    role_order = {
        "operational_baseline": 0,
        "general_budget_anchor": 1,
        "all_month_comparator": 2,
        "pressure_safe_main_candidate": 3,
    }
    scope_order = {"all_available_months": 0, "holdout_pressure": 1, "non_pressure_available_months": 2}
    out["_scope_order"] = out["scope"].map(scope_order)
    out["_role_order"] = out["paper_role"].map(role_order).fillna(0)
    out = out.sort_values(["_scope_order", "_role_order"]).drop(columns=["_scope_order", "_role_order"])
    return out


def build_candidate_decision() -> pd.DataFrame:
    c77 = pd.read_csv(C77_SCREEN)
    promoted = c77[c77["promotable"].eq(True)].copy()
    selected = promoted.iloc[0] if not promoted.empty else None
    boundary = pd.read_csv(C76_BOUNDARY)
    rows = [
        {
            "item": "main_candidate_after_c78",
            "decision": "C77 pressure-safe stress-gated candidate",
            "basis": "Passes validation, holdout pressure, and all-month gates; improves pressure GateScore vs C50 under month-block bootstrap.",
        },
        {
            "item": "c50_role_after_c78",
            "decision": "all-month stronger comparator, not pressure-safe main candidate",
            "basis": "C50 retains better all-month and non-pressure scores, but pressure gain over C44 is weaker.",
        },
        {
            "item": "c44_role_after_c78",
            "decision": "general-budget robust anchor",
            "basis": "C75/C76 show C44 remains necessary for multi-budget stability.",
        },
    ]
    if selected is not None:
        rows.append(
            {
                "item": "selected_c77_rule",
                "decision": str(selected["method"]),
                "basis": (
                    f"val delta {float(selected['val_delta_gate_vs_c44']):.4f}; "
                    f"pressure delta {float(selected['holdout_delta_gate_vs_c44']):.4f}; "
                    f"all-month delta {float(selected['all_delta_gate_vs_c44']):.4f}."
                ),
            }
        )
    for _, r in boundary.iterrows():
        rows.append({"item": r["question"], "decision": r["answer"], "basis": r["experimental_basis"]})
    return pd.DataFrame(rows)


def build_budget_curve_export() -> pd.DataFrame:
    curve = pd.read_csv(C75_CURVE)
    keep = [
        "Base distributional tail-risk ranking",
        "C44 robust conservative anchor",
        "C50 deep-sequence enhanced candidate",
    ]
    out = curve[curve["display_name"].isin(keep)].copy()
    out["display_name"] = out["display_name"].replace(
        {"C50 deep-sequence enhanced candidate": "C50 all-month stronger comparator"}
    )
    return out


def main() -> None:
    main_results = build_main_results()
    candidate_decision = build_candidate_decision()
    budget_curve = build_budget_curve_export()
    boot = pd.read_csv(C78_BOOT)

    main_results.to_csv(OUT / "c79_main_results_with_c77.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "c79_c77_c50_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    candidate_decision.to_csv(OUT / "c79_candidate_decision_after_c78.csv", index=False, encoding="utf-8-sig")
    budget_curve.to_csv(OUT / "c79_budget_curve_c44_c50.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C79 C77 Paper-Ready Integration",
        "",
        "Purpose: integrate C77 into the official paper-ready result structure after C75-C78.",
        "",
        "## Candidate Decision",
        "",
        candidate_decision.to_markdown(index=False),
        "",
        "## Main Results With C77",
        "",
        main_results.to_markdown(index=False),
        "",
        "## Bootstrap Summary",
        "",
        boot.to_markdown(index=False),
        "",
    ]
    (OUT / "c79_c77_paper_ready_integration_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
