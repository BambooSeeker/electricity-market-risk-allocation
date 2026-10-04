from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _near(value: float, expected: float, tol: float = 5e-4) -> bool:
    return abs(value - expected) <= tol


def check_section5_canonical_metrics() -> list[str]:
    path = ROOT / "outputs" / "section5_rebuild_20260611" / "hard_repair_all_canonical_model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    checks: list[str] = []

    targets = {
        "HGB-isotonic tail ranking": {
            "All Recall": 0.7548,
            "All ExcessShare": 0.8313,
            "Pressure Recall": 0.6984,
            "Pressure ExcessShare": 0.7853,
        },
        "Quantile-MLP boundary reranking": {
            "All Recall": 0.7612,
            "All ExcessShare": 0.8498,
            "Pressure Recall": 0.7007,
            "Pressure ExcessShare": 0.7923,
        },
        "Net-load-gated Quantile-MLP boundary replacement": {
            "All Recall": 0.7612,
            "All ExcessShare": 0.8461,
            "Pressure Recall": 0.7053,
            "Pressure ExcessShare": 0.7999,
        },
        "Conditional diffusion-isotonic blend ranking": {
            "All Recall": 0.7500,
            "All ExcessShare": 0.8341,
            "Pressure Recall": 0.6868,
            "Pressure ExcessShare": 0.7779,
        },
    }

    by_name = {row["model_name"]: row for row in rows}
    for name, expected_values in targets.items():
        if name not in by_name:
            raise AssertionError(f"Missing canonical row: {name}")
        row = by_name[name]
        for key, expected in expected_values.items():
            value = _float(row, key)
            if not _near(value, expected):
                raise AssertionError(f"{name} {key}: got {value}, expected {expected}")
        checks.append(f"OK canonical metrics: {name}")
    return checks


def check_nyiso_direct_transfer_metrics() -> list[str]:
    path = ROOT / "outputs" / "cross_market_direct_transfer_20260615" / "zhejiang_to_nyiso_direct_common_feature_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    by_case = {row["case"]: row for row in rows}
    case = "NYISO confirmation pooled"
    if case not in by_case:
        raise AssertionError(f"Missing NYISO pooled case in {path}")
    row = by_case[case]
    expected = {
        "n": 291918,
        "events": 9084,
        "alerts": 87615,
        "recall": 0.8446,
        "excess_share": 0.8133,
        "false_alert_burden": 0.9124,
        "gate_score": 1.4754,
    }
    for key, exp in expected.items():
        value = float(row[key])
        tol = 0.5 if key in {"n", "events", "alerts"} else 5e-4
        if abs(value - exp) > tol:
            raise AssertionError(f"{case} {key}: got {value}, expected {exp}")
    return ["OK NYISO direct common-feature transfer metrics"]


def main() -> None:
    checks = []
    checks.extend(check_section5_canonical_metrics())
    checks.extend(check_nyiso_direct_transfer_metrics())
    print("\n".join(checks))
    print("All packaged result checks passed.")


if __name__ == "__main__":
    main()
