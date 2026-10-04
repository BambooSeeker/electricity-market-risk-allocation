from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c43_budget_aware_boundary_correction import build_alerts, prepare_scores
from c46_validation_safe_modern_complement_selection import PRED
from c49_pressure_transfer_safe_candidate_selection import candidate_alert
from c77_stress_gated_deep_boundary_enhancement import (
    C50,
    stress_gated_alert,
    threshold_from_validation,
)


OUT = ROOT / "manuscript_ready_artifacts" / "case_analysis_upgrade"
OUT.mkdir(parents=True, exist_ok=True)

TABLES = ROOT / "manuscript_ready_artifacts" / "tables"

BUDGET = 0.30
STRESS_FEATURE = "net_load_pred"
STRESS_Q = 0.60
REPLACE_FRAC = 0.05
CANDIDATE_POOL = 0.50


METHOD_ORDER = [
    "Base risk ranking",
    "Boundary-preserved risk screening",
    "Ungated sequence-complement screening",
    "Stress-gated boundary replacement",
]


def manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_scored() -> pd.DataFrame:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    scored = prepare_scores(df)
    scored["hour"] = scored["time"].dt.hour
    return scored


def build_pressure_hour_heatmap(scored: pd.DataFrame) -> None:
    pivot = (
        scored.assign(month=scored["test_month"].astype(str))
        .pivot_table(index="month", columns="hour", values="negative_tail", aggfunc="mean")
        .fillna(0.0)
        * 100
    )
    month_labels = list(pivot.index)

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=180)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(len(month_labels)))
    ax.set_yticklabels(month_labels)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Evaluation month")
    ax.set_title("Lower-tail event intensity by month and hour")

    for y, m in enumerate(month_labels):
        if m in {"2025-09", "2025-10"}:
            ax.add_patch(plt.Rectangle((-0.5, y - 0.5), 24, 1, fill=False, edgecolor="#111111", linewidth=1.5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Event rate (%)")
    fig.tight_layout()
    fig.savefig(OUT / "F10_pressure_event_intensity_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def build_component_delta_matrix() -> None:
    df = pd.read_csv(TABLES / "T1_main_fixed_budget_top30_results.csv")
    scopes = ["All months", "Pressure months", "Non-pressure months"]
    transitions = ["Boundary\nvs Base", "Ungated\nvs Boundary", "Stress-gated\nvs Ungated"]
    rows = []
    for scope in scopes:
        sub = df[df["Scope"].eq(scope)].set_index("Method").loc[METHOD_ORDER]
        vals = sub[["Recall", "ExcessShare", "GateScore"]].to_numpy()
        delta = vals[1:] - vals[:-1]
        for transition, d in zip(transitions, delta):
            rows.append(
                {
                    "Scope": scope,
                    "Transition": transition,
                    "DeltaRecall": d[0],
                    "DeltaExcessShare": d[1],
                    "DeltaGateScore": d[2],
                }
            )
    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(OUT / "component_transition_delta_matrix.csv", index=False, encoding="utf-8-sig")

    mat = delta_df.pivot(index="Scope", columns="Transition", values="DeltaGateScore").loc[scopes, transitions]
    vmax = max(abs(mat.min().min()), abs(mat.max().max()))

    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=180)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions)
    ax.set_yticks(range(len(scopes)))
    ax.set_yticklabels(scopes)
    ax.set_title("Incremental GateScore contribution by screening step")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            row = delta_df[(delta_df["Scope"].eq(scopes[i])) & (delta_df["Transition"].eq(transitions[j]))].iloc[0]
            label = f"G {row['DeltaGateScore']:+.3f}\nR {row['DeltaRecall']:+.3f}\nE {row['DeltaExcessShare']:+.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta GateScore")
    fig.tight_layout()
    fig.savefig(OUT / "F11_component_transition_delta_matrix.png", bbox_inches="tight")
    plt.close(fig)


def build_replacement_risk_map(scored: pd.DataFrame) -> None:
    base_alerts = build_alerts(scored, BUDGET)
    ungated = candidate_alert(scored, C50, BUDGET)
    threshold = threshold_from_validation(scored, STRESS_FEATURE, STRESS_Q)
    stress_gated = stress_gated_alert(
        scored,
        feature=STRESS_FEATURE,
        threshold=threshold,
        replace_frac=REPLACE_FRAC,
        candidate_pool_pct=CANDIDATE_POOL,
    )

    rows = []
    for month in ["2025-09", "2025-10"]:
        idx = scored[scored["test_month"].astype(str).eq(month)].index
        stress_set = set(idx[stress_gated.loc[idx].to_numpy()])
        ungated_set = set(idx[ungated.loc[idx].to_numpy()])
        for group, chosen in [
            ("Added by stress gate", sorted(stress_set - ungated_set)),
            ("Dropped from ungated", sorted(ungated_set - stress_set)),
        ]:
            if not chosen:
                continue
            g = scored.loc[chosen, ["time", STRESS_FEATURE, "negative_excess", "negative_tail", "c22_deep_w20", "base_rank_pct"]].copy()
            g["month"] = month
            g["group"] = group
            rows.append(g)
    changes = pd.concat(rows, ignore_index=True)
    changes.to_csv(OUT / "stress_gated_changed_interval_points.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=180, sharey=True)
    markers = {"Added by stress gate": "o", "Dropped from ungated": "X"}
    colors = {"Added by stress gate": "#b85c38", "Dropped from ungated": "#4b5563"}
    for ax, month in zip(axes, ["2025-09", "2025-10"]):
        sub = changes[changes["month"].eq(month)]
        for group, g in sub.groupby("group"):
            size = 42 + 180 * g["negative_tail"].astype(float)
            ax.scatter(
                g[STRESS_FEATURE],
                g["negative_excess"],
                s=size,
                marker=markers[group],
                color=colors[group],
                alpha=0.78,
                label=group,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.axhline(0, color="#111111", linewidth=0.8)
        ax.set_title(month)
        ax.set_xlabel("Predicted net load")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Realized negative excess")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Changed intervals under stress-gated replacement", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "F12_stress_gated_changed_interval_map.png", bbox_inches="tight")
    plt.close(fig)


def build_budget_gate_heatmap() -> None:
    src = ROOT / "taskB_main_result_strengthening" / "c79_c77_paper_ready_integration" / "c79_budget_curve_c44_c50.csv"
    df = pd.read_csv(src)
    df = df[df["display_name"].isin(
        [
            "Base distributional tail-risk ranking",
            "C44 robust conservative anchor",
            "C50 all-month stronger comparator",
        ]
    )].copy()
    label_map = {
        "Base distributional tail-risk ranking": "Base",
        "C44 robust conservative anchor": "Boundary",
        "C50 all-month stronger comparator": "Ungated",
    }
    scope_map = {
        "all_available_months": "All",
        "holdout_pressure": "Pressure",
        "non_pressure_available_months": "Non-pressure",
    }
    df["Method"] = df["display_name"].map(label_map)
    df["Scope"] = df["scope"].map(scope_map)
    df["Budget"] = (100 * df["budget"]).round().astype(int).astype(str) + "%"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), dpi=180, constrained_layout=True)
    for ax, scope in zip(axes, ["All", "Pressure", "Non-pressure"]):
        sub = df[df["Scope"].eq(scope)]
        mat = sub.pivot_table(index="Method", columns="Budget", values="gate_score", aggfunc="mean")
        mat = mat.loc[["Base", "Boundary", "Ungated"]]
        mat = mat.reindex(columns=sorted(mat.columns, key=lambda s: int(s.rstrip("%"))))
        im = ax.imshow(mat.values, cmap="YlGnBu", aspect="auto")
        ax.set_title(scope)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("GateScore")
    fig.suptitle("Budget-response surface of retained screening families", y=1.08)
    fig.savefig(OUT / "F13_budget_response_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manuscript_style()
    scored = load_scored()
    build_pressure_hour_heatmap(scored)
    build_component_delta_matrix()
    build_replacement_risk_map(scored)
    build_budget_gate_heatmap()


if __name__ == "__main__":
    main()
