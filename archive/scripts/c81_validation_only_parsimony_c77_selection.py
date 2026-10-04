from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
C77_SCREEN = ROOT / "taskB_main_result_strengthening" / "c77_stress_gated_deep_boundary_enhancement" / "c77_candidate_screen.csv"
C80 = ROOT / "taskB_main_result_strengthening" / "c80_c77_selection_leakage_and_ablation_audit"
OUT = ROOT / "taskB_main_result_strengthening" / "c81_validation_only_parsimony_c77_selection"
OUT.mkdir(parents=True, exist_ok=True)


def feature(method: str) -> str:
    body = method.removeprefix("C77_")
    for marker in ["_q60_", "_q70_", "_q80_"]:
        if marker in body:
            return body.split(marker)[0]
    return body


def parse(method: str, key: str) -> int:
    import re

    m = re.search(rf"_{key}(\d+)(?:_|$)", method)
    return int(m.group(1)) if m else -1


def main() -> None:
    df = pd.read_csv(C77_SCREEN)
    df["feature"] = df["method"].map(feature)
    df["q"] = df["method"].map(lambda x: parse(x, "q"))
    df["replace_pct"] = df["method"].map(lambda x: parse(x, "r"))
    df["pool_pct"] = df["method"].map(lambda x: parse(x, "pool"))

    # Clean selection rule:
    # 1) use validation-only gates;
    # 2) restrict to the smallest intervention family (r5, pool50);
    # 3) require nonnegative validation recall and excess gains;
    # 4) select the smallest positive validation GateScore gain, not the largest.
    # This deliberately avoids holdout ranking and avoids validation over-optimization.
    candidates = df[
        df["passes_validation_gate"].eq(True)
        & df["replace_pct"].eq(5)
        & df["pool_pct"].eq(50)
        & (df["val_delta_recall_vs_c44"] >= 0)
        & (df["val_delta_excess_vs_c44"] >= 0)
        & (df["val_delta_gate_vs_c44"] > 0)
    ].copy()
    selected = candidates.sort_values(
        ["val_delta_gate_vs_c44", "q", "feature"], ascending=[True, True, True]
    ).head(1)

    validation_rank = df.sort_values("val_delta_gate_vs_c44", ascending=False).reset_index(drop=True)
    validation_rank["validation_rank_desc"] = validation_rank.index + 1
    selected_method = str(selected.iloc[0]["method"]) if not selected.empty else ""
    selected_rank = validation_rank[validation_rank["method"].eq(selected_method)] if selected_method else pd.DataFrame()

    decision = pd.DataFrame(
        [
            {
                "decision": "PROMOTE_C77_WITH_VALIDATION_ONLY_PARSIMONY_RULE" if selected_method else "NO_GO_C77",
                "selected_method": selected_method,
                "selection_rule": "validation_pass + r5_pool50 + smallest_positive_validation_gate_delta",
                "uses_holdout_for_selection": False,
                "basis": (
                    f"Selected candidate validation delta={float(selected.iloc[0]['val_delta_gate_vs_c44']):.4f}, "
                    f"holdout delta={float(selected.iloc[0]['holdout_delta_gate_vs_c44']):.4f}, "
                    f"all-month delta={float(selected.iloc[0]['all_delta_gate_vs_c44']):.4f}."
                    if selected_method
                    else "No candidate selected."
                ),
            }
        ]
    )

    candidates.to_csv(OUT / "c81_validation_only_candidate_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "c81_selected_validation_only_candidate.csv", index=False, encoding="utf-8-sig")
    selected_rank.to_csv(OUT / "c81_selected_validation_rank.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "c81_validation_only_selection_decision.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C81 Validation-Only Parsimony C77 Selection",
        "",
        "Purpose: rebuild C77 selection without holdout ranking.",
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False),
        "",
        "## Selected Candidate",
        "",
        selected.to_markdown(index=False),
        "",
        "## Candidate Pool",
        "",
        candidates.sort_values("val_delta_gate_vs_c44").to_markdown(index=False),
        "",
    ]
    (OUT / "c81_validation_only_parsimony_c77_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
