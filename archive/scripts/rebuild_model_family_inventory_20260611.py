from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'model_family_rebuild_20260611'
OUT.mkdir(exist_ok=True)
BUDGET = 0.30
PRESSURE_MONTHS = {'2025-09', '2025-10'}
EVAL_MONTHS = {'2025-09','2025-10','2025-11','2025-12','2026-01','2026-02'}

def display_float(x):
    if pd.isna(x): return ''
    return f'{x:.4f}'

def metrics_from_prediction(path, score_cols, family, role, source, notes='', months_filter=EVAL_MONTHS):
    p = ROOT / path
    rows=[]
    if not p.exists():
        return rows
    df = pd.read_csv(p, parse_dates=['time'] if 'time' in pd.read_csv(p, nrows=0).columns else None)
    if 'test_month' in df.columns:
        df['test_month'] = df['test_month'].astype(str)
        df = df[df['test_month'].isin(months_filter)].copy()
    if 'negative_excess' not in df.columns:
        df['negative_excess'] = 0.0
    if 'negative_tail' not in df.columns:
        return rows
    for method_name, col in score_cols:
        if col not in df.columns:
            continue
        for scope, g in [('all_available_months', df), ('holdout_pressure', df[df['test_month'].isin(PRESSURE_MONTHS)]), ('non_pressure_available_months', df[~df['test_month'].isin(PRESSURE_MONTHS)])]:
            if g.empty:
                continue
            alerts = pd.Series(False, index=g.index)
            if 'test_month' in g.columns:
                for _, gm in g.groupby('test_month'):
                    k = int(np.ceil(BUDGET * len(gm)))
                    idx = gm[col].astype(float).sort_values(ascending=False).index[:k]
                    alerts.loc[idx] = True
            else:
                k = int(np.ceil(BUDGET * len(g)))
                idx = g[col].astype(float).sort_values(ascending=False).index[:k]
                alerts.loc[idx] = True
            y = g['negative_tail'].astype(int)
            ex = g['negative_excess'].astype(float)
            n_events = int(y.sum())
            n_alerts = int(alerts.sum())
            hits = int((alerts & (y == 1)).sum())
            precision = hits / n_alerts if n_alerts else np.nan
            recall = hits / n_events if n_events else np.nan
            excess_share = float(ex[alerts].sum() / ex.sum()) if ex.sum() > 0 else np.nan
            fab = float(((alerts) & (y == 0)).sum() / n_alerts) if n_alerts else np.nan
            gate = recall + excess_share - 0.2 * fab if not any(pd.isna([recall, excess_share, fab])) else np.nan
            rows.append({
                'model_family': family,
                'model_name': method_name,
                'role': role,
                'source_code_or_output': source,
                'scope': scope,
                'budget': BUDGET,
                'n': len(g),
                'events': n_events,
                'alerts': n_alerts,
                'precision': precision,
                'recall': recall,
                'excess_share': excess_share,
                'false_alert_burden': fab,
                'gate_score': gate,
                'evaluation_window': ','.join(sorted(g['test_month'].unique())) if 'test_month' in g.columns else '',
                'main_text_eligibility': 'eligible_same_top30_protocol' if set(g['test_month'].unique()).issubset(EVAL_MONTHS) else 'audit_only_window_mismatch',
                'notes': notes,
            })
    return rows

def metrics_from_metric_file(path, method_map, family, role, source, notes='', preferred_budget=BUDGET):
    p = ROOT / path
    rows=[]
    if not p.exists(): return rows
    df = pd.read_csv(p)
    if 'budget' in df.columns:
        df = df[np.isclose(df['budget'].astype(float), preferred_budget)].copy()
    for raw, name in method_map.items():
        h = df[df['method'].astype(str)==raw].copy() if 'method' in df.columns else pd.DataFrame()
        for _, r in h.iterrows():
            rows.append({
                'model_family': family,
                'model_name': name,
                'role': role,
                'source_code_or_output': source,
                'scope': r.get('scope',''),
                'budget': r.get('budget', preferred_budget),
                'n': r.get('n',''),
                'events': r.get('events',''),
                'alerts': r.get('alerts',''),
                'precision': r.get('precision',np.nan),
                'recall': r.get('recall',np.nan),
                'excess_share': r.get('excess_share',np.nan),
                'false_alert_burden': r.get('false_alert_burden',np.nan),
                'gate_score': r.get('gate_score',np.nan),
                'evaluation_window': '',
                'main_text_eligibility': 'eligible_same_top30_protocol' if 'all_available_months' in str(r.get('scope','')) or 'holdout_pressure' in str(r.get('scope','')) or 'non_pressure' in str(r.get('scope','')) else 'audit_only_scope_check',
                'notes': notes,
            })
    return rows

rows=[]
# Current paper-ready core and internal methods.
pr = ROOT/'paper_ready_results'/'baseline_table.csv'
if pr.exists():
    df = pd.read_csv(pr)
    for _, r in df.iterrows():
        for scope, prefix in [('all_available_months','all'),('holdout_pressure','pressure')]:
            rows.append({
                'model_family': r['family'],
                'model_name': r['display_name'],
                'role': r['paper_role'],
                'source_code_or_output': r['source'] + ' / paper_ready_results/baseline_table.csv',
                'scope': scope,
                'budget': BUDGET,
                'n': '', 'events': '', 'alerts': '', 'precision': np.nan,
                'recall': r[f'{prefix}_recall'],
                'excess_share': r[f'{prefix}_excess_share'],
                'false_alert_burden': np.nan,
                'gate_score': r[f'{prefix}_gate_score'],
                'evaluation_window': '2025-09..2026-02',
                'main_text_eligibility': 'eligible_same_top30_protocol',
                'notes': r.get('claim_role',''),
            })

# Standalone quantile deep-learning prediction files.
rows += metrics_from_prediction('qia_tail_lstm_quantile_module/lstm_quantile_predictions.csv', [
    ('LSTM quantile probability ranking','calibrated_probability'),
    ('LSTM quantile risk ranking','lstm_qia_raw'),
], 'Standalone recurrent deep quantile model', 'External deep-learning baseline', 'qia_tail_lstm_quantile_module.py')
rows += metrics_from_prediction('qia_tail_tcn_quantile_module/tcn_quantile_predictions.csv', [
    ('TCN quantile probability ranking','calibrated_probability'),
    ('TCN quantile risk ranking','tcn_qia_raw'),
], 'Standalone temporal-convolution quantile model', 'External deep-learning baseline', 'qia_tail_tcn_quantile_module.py')
rows += metrics_from_prediction('qia_tail_transformer_quantile_module/transformer_quantile_predictions.csv', [
    ('Transformer quantile probability ranking','calibrated_probability'),
    ('Transformer quantile risk ranking','transformer_qia_raw'),
], 'Standalone Transformer quantile model', 'External deep-learning baseline', 'qia_tail_transformer_quantile_module.py')

# Stronger later deep sequence baselines from metric files.
rows += metrics_from_metric_file('c60_dilated_tcn_attention_tail_baseline/c60_dilated_tcn_attention_metrics.csv', {
    'c60_tcn_tail_prob':'Dilated TCN attention tail-probability ranking',
    'c60_tcn_q05_risk':'Dilated TCN attention q05-risk ranking',
}, 'Dilated TCN attention model', 'External deep-learning baseline', 'c60_dilated_tcn_attention_tail_baseline.py')
rows += metrics_from_metric_file('c61_pressure_aware_tcn_attention_tail_baseline/c61_pressure_aware_tcn_attention_metrics.csv', {
    'c61_tcn_tail_prob':'Pressure-aware TCN attention tail-probability ranking',
    'c61_tcn_tail_prob_iso':'Pressure-aware TCN attention isotonic tail ranking',
}, 'Pressure-aware TCN attention model', 'External deep-learning baseline', 'c61_pressure_aware_tcn_attention_tail_baseline.py')
rows += metrics_from_metric_file('c59_local_selective_ssm_tail_baseline/c59_local_selective_ssm_metrics.csv', {
    'c59_selssm_tail_prob':'Selective SSM tail-probability ranking',
    'c59_selssm_q05_risk':'Selective SSM q05-risk ranking',
}, 'Selective state-space sequence model', 'External sequence baseline', 'c59_local_selective_ssm_tail_baseline.py')

# Diffusion standalone / generative baselines.
rows += metrics_from_metric_file('c33_conditional_diffusion_residual_baseline/c33_conditional_diffusion_metrics.csv', {
    'c33_diff_tail_prob':'Conditional diffusion tail-probability ranking',
    'c33_diff_expected_shortfall':'Conditional diffusion expected-shortfall ranking',
    'c33_diff_q20_risk':'Conditional diffusion q20-risk ranking',
}, 'Conditional diffusion scenario model', 'External diffusion baseline', 'c33_conditional_diffusion_residual_baseline.py')
rows += metrics_from_prediction('tail_aware_residual_diffusion_outputs/tail_aware_residual_diffusion_predictions.csv', [
    ('Tail-aware residual diffusion probability ranking','TailAwareDiffusion_p_negative'),
    ('Tail-aware residual diffusion expected-shortfall ranking','TailAwareDiffusion_expected_shortfall'),
], 'Tail-aware residual diffusion model', 'External diffusion baseline', 'tail_aware_residual_diffusion_slts_remote.py', notes='starts at 2025-10; pressure slice excludes September')
rows += metrics_from_prediction('qia_tail_diffusion_distribution_module/diffusion_qia_predictions.csv', [
    ('Diffusion-QIA tail-probability ranking','diffusion_tail_probability'),
    ('Diffusion-QIA expected-negative-excess ranking','diffusion_expected_negative_excess'),
    ('Diffusion-QIA q10-distance ranking','diffusion_neg_distance_q10'),
], 'Diffusion distribution module', 'External diffusion baseline', 'qia_tail_diffusion_distribution_module.py')

# Advanced deep rank and fusion diagnostics.
rows += metrics_from_prediction('advanced_deep_rank_outputs/advanced_deep_rank_predictions.csv', [
    ('Focal-pair deep ranking, two-stage score','score_two_stage'),
    ('Focal-pair deep ranking, hybrid score','score_hybrid'),
], 'Focal-pair deep ranking model', 'External deep-learning diagnostic', 'advanced_deep_rank_slts_remote.py')
rows += metrics_from_prediction('unified_signal_stacking_experiment/unified_stacking_predictions.csv', [
    ('Tree-only logistic stacking','Stack_TreeOnly_plain'),
    ('Tree plus advanced-deep logistic stacking','Stack_TreeAdvDeep_plain'),
    ('Tree-deep-diffusion logistic stacking','Stack_TreeDeepDiff_plain'),
    ('Tree-deep-diffusion severity-weighted stacking','Stack_TreeDeepDiff_severity'),
], 'Signal stacking model', 'Supplementary fusion diagnostic', 'unified_signal_stacking_experiment.py', notes='stacking starts at 2025-12; not eligible for pressure-month claims')

out = pd.DataFrame(rows)
out.to_csv(OUT/'model_family_inventory_long.csv', index=False, encoding='utf-8-sig')

# Build recommended Section 5 candidate table: one representative per family/model, all and pressure columns.
eligible = out[out['main_text_eligibility'].eq('eligible_same_top30_protocol')].copy()
# Prefer concise selected candidates; include all rows where all or pressure available.
selected_names = [
    'HGB-isotonic tail ranking',
    'DA-congestion full ranking',
    'Quantile-MLP post-processing',
    'Conditional diffusion q20-risk ranking',
    'Dilated TCN attention tail-probability ranking',
    'Pressure-aware TCN attention tail-probability ranking',
    'Selective SSM tail-probability ranking',
    'Core-preserved fixed-budget screening',
    'Quantile-MLP boundary reranking',
    'Net-load-gated Quantile-MLP replacement',
]
sel = eligible[eligible['model_name'].isin(selected_names)].copy()
wide=[]
for (fam,name,role,src), g in sel.groupby(['model_family','model_name','role','source_code_or_output'], dropna=False):
    rec={'model_family':fam,'model_name':name,'role':role,'source':src}
    for scope,label in [('all_available_months','all'),('holdout_pressure','pressure')]:
        h=g[g['scope'].eq(scope)]
        if not h.empty:
            r=h.iloc[0]
            rec[f'{label}_recall']=r['recall']; rec[f'{label}_excess_share']=r['excess_share']; rec[f'{label}_gate_score']=r['gate_score']
    wide.append(rec)
wide=pd.DataFrame(wide)
if not wide.empty:
    wide=wide.sort_values(['role','model_family','model_name'])
wide.to_csv(OUT/'section5_candidate_model_family_table.csv', index=False, encoding='utf-8-sig')

# Audit-only table summarizing non-main candidates or window mismatches.
audit = out.copy()
audit['is_selected_main_candidate'] = audit['model_name'].isin(selected_names)
audit.to_csv(OUT/'model_family_full_audit_table.csv', index=False, encoding='utf-8-sig')

# Markdown summary.
with open(OUT/'model_family_rebuild_summary.md','w',encoding='utf-8') as f:
    f.write('# Model family rebuild summary, 2026-06-11\n\n')
    f.write('This file is generated from code outputs. It separates external baselines, internal components, retained variants, standalone deep-learning models, standalone diffusion models, and fusion diagnostics.\n\n')
    f.write('## Recommended Section 5 comparison table shape\n\n')
    f.write('The formal table should use `Model family`, `Model name`, `Role`, and then metrics. The first two columns must not be merged or replaced by broad labels.\n\n')
    if not wide.empty:
        f.write(wide.to_markdown(index=False, floatfmt='.4f'))
    f.write('\n\n## Important cautions\n\n')
    f.write('- Standalone LSTM, TCN, Transformer, TCN-attention, SSM, and diffusion models exist and must not be collapsed into one name.\n')
    f.write('- `Tree-deep-diffusion logistic stacking` exists, but it starts at 2025-12 and cannot support September-October pressure-period claims.\n')
    f.write('- Tail-aware residual diffusion starts at 2025-10, so it is useful as a diffusion audit but not as a complete September-February pressure comparison unless rerun.\n')
    f.write('- Main text should use only methods with the same fixed-budget top30 protocol and clearly stated evaluation window.\n')
    f.write('- Section 5 must distinguish the strong empirical finding, fixed-budget concentration, from the narrower incremental finding, pressure-conditioned boundary replacement.\n')

print('WROTE', OUT)
print('candidate table:')
print(wide.to_string(index=False))
