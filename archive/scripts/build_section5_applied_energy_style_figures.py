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
BOOTSTRAP_FULL = (
    ROOT
    / "taskB_main_result_strengthening"
    / "c79_c77_paper_ready_integration"
    / "c79_c77_c50_bootstrap_summary.csv"
)
BUDGET_CURVE = (
    ROOT
    / "taskB_main_result_strengthening"
    / "c79_c77_paper_ready_integration"
    / "c79_budget_curve_c44_c50.csv"
)

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

BLUE = "#0072BD"
ORANGE = "#D95319"
YELLOW = "#EDB120"
PURPLE = "#7E2F8E"
GREEN = "#77AC30"
CYAN = "#4DBEEE"
RED = "#A2142F"
GRAY = "#808080"
BLACK = "#000000"
LIGHT_GRAY = "#BFBFBF"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.75,
            "axes.edgecolor": BLACK,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": BLACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
        }
    )


def load_scored() -> pd.DataFrame:
    df = pd.read_csv(PRED, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    scored = prepare_scores(df)
    scored["hour"] = scored["time"].dt.hour
    scored["month"] = scored["test_month"].astype(str)
    return scored


def fig_case_profile(scored: pd.DataFrame) -> None:
    monthly = pd.read_csv(OUT / "pressure_period_monthly_profile.csv")
    monthly["month"] = monthly["month"].astype(str)

    hour_rate = (
        scored.groupby(["month", "hour"])["negative_tail"]
        .mean()
        .unstack("month")
        .sort_index()
        * 100
    )

    pressure = monthly[monthly["slice"].eq("Pressure")]
    non = monthly[monthly["slice"].eq("Non-pressure")]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), dpi=200)
    ax = axes[0, 0]
    colors = [ORANGE if s == "Pressure" else BLUE for s in monthly["slice"]]
    ax.bar(monthly["month"], 100 * monthly["lower_tail_event_rate"], color=colors, edgecolor=BLACK, linewidth=0.35, width=0.65)
    ax.set_ylabel("Event rate (%)")
    ax.set_xlabel("Month")
    ax.set_title("(a) Lower-tail event frequency")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    ax.plot(monthly["month"], monthly["abs_spread_q95"], marker="o", color=BLUE, label="Abs. spread q95")
    ax.plot(monthly["month"], monthly["spread_std"], marker="s", color=ORANGE, label="Spread std.")
    ax.set_ylabel("Spread magnitude")
    ax.set_xlabel("Month")
    ax.set_title("(b) Spread volatility and upper tail")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    month_colors = {"2025-09": BLUE, "2025-10": ORANGE, "2025-11": GREEN, "2026-02": RED}
    for col in ["2025-09", "2025-10", "2025-11", "2026-02"]:
        if col in hour_rate:
            lw = 1.8 if col in {"2025-09", "2025-10"} else 1.0
            style = "-" if col in {"2025-09", "2025-10"} else "--"
            ax.plot(hour_rate.index, hour_rate[col], marker="o", markersize=2.8, linewidth=lw, linestyle=style, color=month_colors[col], label=col)
    ax.set_ylabel("Event rate (%)")
    ax.set_xlabel("Hour")
    ax.set_title("(c) Intraday event profile")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[1, 1]
    labels = ["Pressure", "Non-pressure"]
    event_rates = [100 * pressure["lower_tail_event_rate"].mean(), 100 * non["lower_tail_event_rate"].mean()]
    q95 = [pressure["abs_spread_q95"].mean(), non["abs_spread_q95"].mean()]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, event_rates, width, color=ORANGE, edgecolor=BLACK, linewidth=0.35, label="Event rate (%)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, q95, width, color=BLUE, edgecolor=BLACK, linewidth=0.35, label="Abs. spread q95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Event rate (%)")
    ax2.set_ylabel("Abs. spread q95")
    ax.set_title("(d) Pressure vs non-pressure contrast")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "F10_ae_case_profile_multipanel.png", bbox_inches="tight")
    plt.close(fig)


def fig_component_scenario_chain() -> None:
    df = pd.read_csv(TABLES / "T1_main_fixed_budget_top30_results.csv")
    label_map = {
        "Base risk ranking": "Base",
        "Boundary-preserved risk screening": "Boundary",
        "Ungated sequence-complement screening": "Ungated",
        "Stress-gated boundary replacement": "Stress-gated",
    }
    df["MethodLabel"] = df["Method"].map(label_map)
    scopes = ["All months", "Pressure months", "Non-pressure months"]
    metrics = [("Recall", "Recall"), ("ExcessShare", "ExcessShare"), ("GateScore", "GateScore")]
    colors = {"All months": BLUE, "Pressure months": ORANGE, "Non-pressure months": GREEN}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=200)
    for ax, (metric, title) in zip(axes, metrics):
        for scope in scopes:
            sub = df[df["Scope"].eq(scope)].set_index("Method").loc[METHOD_ORDER]
            ax.plot(
                [label_map[m] for m in METHOD_ORDER],
                sub[metric],
                marker="o",
                linewidth=1.8,
                label=scope,
                color=colors[scope],
            )
        ax.set_title(f"({chr(97 + metrics.index((metric, title)))}) {title}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "F11_ae_component_scenario_chain.png", bbox_inches="tight")
    plt.close(fig)


def fig_replacement_mechanism_stack(scored: pd.DataFrame) -> None:
    summary = pd.read_csv(OUT / "stress_gated_replacement_mechanism_pressure.csv")
    summary = summary[summary["comparison"].eq("stress_gated_vs_ungated")].copy()

    base_alerts = build_alerts(scored, BUDGET)
    ungated = candidate_alert(scored, C50, BUDGET)
    threshold = threshold_from_validation(scored, STRESS_FEATURE, STRESS_Q)
    gated = stress_gated_alert(
        scored,
        feature=STRESS_FEATURE,
        threshold=threshold,
        replace_frac=REPLACE_FRAC,
        candidate_pool_pct=CANDIDATE_POOL,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), dpi=200)

    ax = axes[0, 0]
    months = ["2025-09", "2025-10"]
    x = np.arange(len(months))
    width = 0.35
    added = summary[summary["group"].eq("added_by_stress_gated")].set_index("month").loc[months]
    dropped = summary[summary["group"].eq("dropped_by_stress_gated")].set_index("month").loc[months]
    ax.bar(x - width / 2, added["negative_excess_sum"], width, color=ORANGE, edgecolor=BLACK, linewidth=0.35, label="Added")
    ax.bar(x + width / 2, dropped["negative_excess_sum"], width, color=GRAY, edgecolor=BLACK, linewidth=0.35, label="Dropped")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("Negative excess sum")
    ax.set_title("(a) Severity carried by changed intervals")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.bar(x - width / 2, 100 * added["tail_event_rate"], width, color=ORANGE, edgecolor=BLACK, linewidth=0.35, label="Added")
    ax.bar(x + width / 2, 100 * dropped["tail_event_rate"], width, color=GRAY, edgecolor=BLACK, linewidth=0.35, label="Dropped")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("Tail-event rate (%)")
    ax.set_title("(b) Event concentration")
    ax.legend(loc="upper left", fontsize=8)

    changed_rows = []
    for month in months:
        idx = scored[scored["month"].eq(month)].index
        gated_set = set(idx[gated.loc[idx].to_numpy()])
        ungated_set = set(idx[ungated.loc[idx].to_numpy()])
        for group, chosen in [
            ("Added", sorted(gated_set - ungated_set)),
            ("Dropped", sorted(ungated_set - gated_set)),
        ]:
            for i in chosen:
                changed_rows.append(
                    {
                        "month": month,
                        "hour": int(scored.loc[i, "hour"]),
                        "group": group,
                        "negative_excess": scored.loc[i, "negative_excess"],
                        "negative_tail": scored.loc[i, "negative_tail"],
                    }
                )
    points = pd.DataFrame(changed_rows)

    ax = axes[1, 0]
    for group, marker, color in [("Added", "o", ORANGE), ("Dropped", "x", GRAY)]:
        sub = points[points["group"].eq(group)]
        ax.scatter(sub["hour"], sub["negative_excess"], marker=marker, color=color, alpha=0.85, label=group)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Negative excess")
    ax.set_title("(c) Intraday location of changed intervals")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    hourly = (
        points.assign(weight=lambda x: np.where(x["group"].eq("Added"), x["negative_excess"], -x["negative_excess"]))
        .groupby("hour")["weight"]
        .sum()
        .reindex(range(24), fill_value=0)
    )
    colors = [ORANGE if v >= 0 else GRAY for v in hourly]
    ax.bar(hourly.index, hourly.values, color=colors, edgecolor=BLACK, linewidth=0.25, width=0.75)
    ax.axhline(0, color=BLACK, linewidth=0.8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Net added severity")
    ax.set_title("(d) Net severity shift by hour")

    fig.tight_layout()
    fig.savefig(OUT / "F12_ae_replacement_mechanism_multipanel.png", bbox_inches="tight")
    plt.close(fig)


def fig_robustness_sensitivity() -> None:
    curve = pd.read_csv(BUDGET_CURVE)
    curve = curve[curve["display_name"].isin(
        [
            "Base distributional tail-risk ranking",
            "C44 robust conservative anchor",
            "C50 all-month stronger comparator",
        ]
    )].copy()
    method_map = {
        "Base distributional tail-risk ranking": "Base",
        "C44 robust conservative anchor": "Boundary",
        "C50 all-month stronger comparator": "Ungated",
    }
    scope_map = {
        "all_available_months": "All months",
        "holdout_pressure": "Pressure months",
        "non_pressure_available_months": "Non-pressure months",
    }
    curve["method"] = curve["display_name"].map(method_map)
    curve["scope_label"] = curve["scope"].map(scope_map)

    boot_src = pd.read_csv(BOOTSTRAP_FULL)
    boot_src = boot_src[boot_src["bootstrap_type"].eq("month_block")].copy()
    comparison_map = {
        "C77_vs_C50": "Stress-gated replacement vs ungated",
        "C77_vs_C44": "Stress-gated replacement vs boundary-preserved",
        "C50_vs_C44": "Ungated vs boundary-preserved",
    }
    scope_map = {
        "all_available_months": "All months",
        "holdout_pressure": "Pressure months",
        "non_pressure_available_months": "Non-pressure months",
    }
    boot_src = boot_src[boot_src["comparison"].isin(comparison_map)].copy()
    boot_src = boot_src[boot_src["scope"].isin(scope_map)].copy()
    boot_src["Comparison"] = boot_src["comparison"].map(comparison_map)
    boot_src["Scope"] = boot_src["scope"].map(scope_map)
    keep_rows = [
        ("C50_vs_C44", "all_available_months"),
        ("C50_vs_C44", "holdout_pressure"),
        ("C77_vs_C44", "all_available_months"),
        ("C77_vs_C44", "holdout_pressure"),
        ("C77_vs_C44", "non_pressure_available_months"),
        ("C77_vs_C50", "all_available_months"),
        ("C77_vs_C50", "holdout_pressure"),
        ("C77_vs_C50", "non_pressure_available_months"),
    ]
    boot = pd.concat(
        [
            boot_src[(boot_src["comparison"].eq(comp)) & (boot_src["scope"].eq(scope))]
            for comp, scope in keep_rows
        ],
        ignore_index=True,
    )
    boot = boot.rename(
        columns={
            "delta_gate_score_mean": "MeanDeltaGateScore",
            "delta_gate_score_p025": "GateDeltaCI2.5",
            "delta_gate_score_p975": "GateDeltaCI97.5",
        }
    )
    boot["label"] = boot["Scope"] + "\n" + boot["Comparison"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=200)

    ax = axes[0]
    for (scope, method), sub in curve.groupby(["scope_label", "method"]):
        if scope != "Pressure months":
            continue
        style = "-" if method == "Ungated" else "--"
        marker = "o" if method == "Ungated" else "s"
        ax.plot(100 * sub["budget"], sub["gate_score"], linestyle=style, marker=marker, linewidth=1.6, label=method)
    ax.set_xlabel("Alert budget (%)")
    ax.set_ylabel("GateScore")
    ax.set_title("(a) Pressure-month budget sensitivity")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    y = np.arange(len(boot))
    x = boot["MeanDeltaGateScore"]
    xerr = [x - boot["GateDeltaCI2.5"], boot["GateDeltaCI97.5"] - x]
    ax.errorbar(x, y, xerr=xerr, fmt="o", color=BLUE, ecolor=GRAY, capsize=3)
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(boot["label"], fontsize=7)
    ax.set_xlabel("Bootstrap GateScore difference")
    ax.set_title("(b) Bootstrap claim boundary")
    ax.grid(True, axis="x", alpha=0.22)

    fig.tight_layout()
    fig.savefig(OUT / "F13_ae_robustness_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    scored = load_scored()
    fig_case_profile(scored)
    fig_component_scenario_chain()
    fig_replacement_mechanism_stack(scored)
    fig_robustness_sensitivity()


if __name__ == "__main__":
    main()
