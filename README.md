# Fixed-budget electricity-market risk allocation

Research software for **Fixed-Budget Lower-Tail Risk Alert Allocation in Day-Ahead and Real-Time Electricity Markets**.

The repository implements protected-core selection, fixed-count boundary exchange, the conditional score–membership frontier, NYISO reconstruction and evaluation, and the capacity-value analysis. Author-owned code is released under BSD-3-Clause. Data retain their providers' terms.

## Installation

Use Python 3.12 with the pinned environment:

```sh
git clone https://github.com/BambooSeeker/electricity-market-risk-allocation.git
cd electricity-market-risk-allocation
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the public analyses

```sh
python -m unittest -v test_allocation.py
python prepare_public_data.py
python reproduce_nyiso.py --historical-only --out results/nyiso_historical
python reproduce_nyiso.py --out results/nyiso
python reproduce_capacity_value.py
```

`prepare_public_data.py` reconstructs observations from the official monthly archives bundled in `data/nyiso/raw`. To fetch missing source files and verify their recorded hashes, run `python download_nyiso.py` first. The provider can revise historical archives; the source manifest identifies the versions used here.

The historical branch is recomputed from public records. The source-score branch uses `data/nyiso/source_score_export.csv`, which contains fitted predictions on the public target observations. Refitting the Zhejiang-source predictor requires access to its restricted training history.

Expected NYISO reconstruction: 96,096 annual matched observations; 48,587 evaluation observations across 11 zones and 66 zone-months, with 1,535 tail events and 14,619 alerts. Frozen historical Recall and ExcessShare are approximately 0.673616 and 0.844188; local adjustment gives 0.671010 and 0.845538. Calendar-month bootstrap blocks retain simultaneous zones, with seed 20261004.

Capacity-value reproduction uses the archived aggregate profile in `results/capacity_value`. It returns marginal normalized break-even costs of approximately 20.27, 10.71, 6.86 and 2.27 CNY per added alert. These values describe the manuscript's budget-specific policies under its stated exposure and response assumptions.

## Data access

### NYISO

- Official market-data portal: https://www.nyiso.com/energy-market-operational-data
- Official CSV archives: https://mis.nyiso.com/public/csv/
- Exact source URLs and SHA-256 hashes: `docs/nyiso_source_manifest.csv`

The repository includes the 36 monthly source archives for 2025, processed observations, target-score exports and evaluation outputs. Hourly matching uses the latest forecast vintage issued before delivery day and explicit occurrence keys for repeated daylight-saving hours. NYISO data remain subject to the provider's terms.

### Zhejiang

Source and access platform: https://zjpx.com.cn/

Zhejiang interval observations are subject to data-provider confidentiality and redistribution requirements. Full records must be requested through the provider, subject to market eligibility and authorization. The authors' selected seven-day sample, 1–7 September 2025, remains local pending data-holder clearance. Its observations, derived interval predictions and saved-score audit are excluded from this public repository.

Paper-level Zhejiang aggregates are provided in `data/summary`, together with the capacity profile and rendered result figures. They support the disclosed aggregate calculations. Original prediction refitting and full-history interval figure reconstruction require authorized observations.

## Allocation and predictive implementations

`risk_allocation.py` implements deterministic selection, protected-core boundary allocation, fixed-count stress exchange and the conditional frontier. Equal scores resolve in input order. Records are ordered chronologically by the evaluation scripts.

The manuscript evaluates monthly-cohort rankings as a retrospective capacity benchmark. Sequential admission with arriving observations requires an explicit remaining-capacity policy.

`archive/scripts` preserves the original authored predictive and analysis implementations, including Quantile-MLP, sequence, diffusion and pretrained-model comparisons. `docs/archived_implementations.md` and `docs/archived_script_manifest.json` identify dependencies and checksums. These implementations retain their original data structures and local path defaults; authorized users should adapt those paths to their own data layout. The minimal public reproduction commands are listed above.

`run_demo.py` provides a chronological HGB training, calibration and evaluation workflow for a separately authorized 336-row half-hourly sample. The first 192 records train, the next 48 calibrate and the remaining 96 test. Its feature subset and short history define a compact demonstration; its metrics have a different scope from the full paper evaluation. `audit_saved_week.py` evaluates an authorized saved-score slice from full monthly selections.

## Figures

```sh
python -m pip install -r requirements-plots.txt
python plot_paper_results.py
```

The command reconstructs Figure 7 from the NYISO outputs. Rendered figures and aggregate summaries are included for inspection. Rebuilding Zhejiang Figures 5–6 requires `python plot_zhejiang_temporal.py --input /path/to/authorized_alert_audit.csv`. The code license does not grant redistribution rights to datasets or manuscript figures.

## Citation and contact

Citation metadata are in `CITATION.cff`; file checksums are in `docs/release_manifest.json`. Version 1.0.0 accompanies the Energy submission manuscript; no article DOI has been assigned.

Corresponding author: Peng Hou, Tianjin University, houpeng2026@tju.edu.cn.
