"""Recompute public NYISO cohort comparisons with dated DA forecast vintages."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from risk_allocation import boundary, coverage, frozen_ecdf, topk


def main(data,out,historical_only=False):
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data,parse_dates=['timestamp','issue_date'])
    df=df.sort_values(['zone','timestamp'],kind='stable').reset_index(drop=True)
    df['dst_occurrence']=df['occurrence'] if 'occurrence' in df else df.groupby(['zone','timestamp']).cumcount()
    if df.duplicated(['zone','timestamp','dst_occurrence']).any():
        raise ValueError('Duplicate settlement identifier')
    if not (df.issue_date.dt.normalize()<df.timestamp.dt.normalize()).all():
        raise ValueError('DA forecast vintage must precede delivery day')
    df['month']=df.timestamp.dt.to_period('M').astype(str)
    validation=df.timestamp.between('2025-04-01','2025-06-30 23:59:59')
    confirm=df.timestamp>pd.Timestamp('2025-06-30 23:59:59')
    # Hour-specific histories are shifted before rolling, matching the clean archive.
    df['hour']=df.timestamp.dt.hour
    shifted=df.groupby(['zone','hour']).da_rt_spread.shift(1)
    events=df.groupby(['zone','hour']).lower_tail_event.shift(1)
    df['historical_q10']=shifted.groupby([df.zone,df.hour]).rolling(60,min_periods=10).quantile(.1).reset_index(level=[0,1],drop=True)
    df['historical_frequency']=events.groupby([df.zone,df.hour]).rolling(30,min_periods=10).mean().reset_index(level=[0,1],drop=True)
    df['base']=np.nan; df['local_signal']=np.nan
    for zone,idx in df.groupby('zone').groups.items():
        ref=df.loc[validation & df.zone.eq(zone)]
        df.loc[idx,'base']=frozen_ecdf(-df.loc[idx,'historical_q10'],-ref.historical_q10)
        df.loc[idx,'local_signal']=frozen_ecdf(df.loc[idx,'historical_frequency'],ref.historical_frequency)
    df['local']=df.base+.2*df.local_signal
    df['source_local']=df.zj_source_score+.2*df.local_signal
    methods=['base','local'] if historical_only else ['base','local','source_base','source_local']
    alerts={key:pd.Series(False,index=df.index) for key in methods}
    for key,g in df[confirm].groupby('zone_month',sort=True):
        idx=g.index
        alerts['base'].loc[idx]=topk(g.base.to_numpy(),.3)
        alerts['local'].loc[idx]=boundary(g.base.to_numpy(),g.local.to_numpy(),.3,.075,.5)[0]
        if historical_only:
            continue
        if g.zj_source_score.notna().all():
            alerts['source_base'].loc[idx]=topk(g.zj_source_score.to_numpy(),.3)
            alerts['source_local'].loc[idx]=boundary(g.zj_source_score.to_numpy(),g.source_local.to_numpy(),.3,.075,.5)[0]
        else:
            raise ValueError(f'Missing source scores in {key}')
    records=[]
    for key,g in df[confirm].groupby('zone_month',sort=True):
        tail=g[g.lower_tail_event.eq(1)]
        row={'zone_month':key,'zone':g.zone.iloc[0],'month':g.month.iloc[0],'n':len(g),
             'events':int(g.lower_tail_event.sum()),'excess':float(g.negative_excess.sum()),
             'event_rate':g.lower_tail_event.mean(),'excess_intensity':g.negative_excess.mean(),
             'volatility':g.da_rt_spread.std(),'tail_congestion':tail.day_ahead_congestion.abs().mean() if len(tail) else 0.,
             'load_mean':g.load_forecast.mean()}
        for method in methods:
            selected=g[alerts[method].loc[g.index].to_numpy()]
            row[method+'_events']=int(selected.lower_tail_event.sum())
            row[method+'_excess']=float(selected.negative_excess.sum())
            row[method+'_alerts']=len(selected)
        records.append(row)
    blocks=pd.DataFrame(records)
    dims=['event_rate','excess_intensity','volatility','tail_congestion']
    blocks['pressure_dimensions']=sum(blocks[c]>=blocks[c].median() for c in dims)
    vload=df[validation].groupby(['zone','zone_month']).load_forecast.mean().groupby('zone').quantile(.6)
    blocks['high_load']=blocks.load_mean>=blocks.zone.map(vload)
    scopes={'all':blocks,'pressure':blocks[blocks.pressure_dimensions>=3],
            'strict_pressure':blocks[blocks.pressure_dimensions>=4],'high_load':blocks[blocks.high_load]}
    summary=[]; audit=[]
    for scope,b in scopes.items():
        comparisons=[('base','local')]
        if scope=='all' and not historical_only:
            comparisons.append(('source_base','source_local'))
        for baseline,local in comparisons:
            def contrast(group):
                br=group[baseline+'_events'].sum()/group.events.sum()
                lr=group[local+'_events'].sum()/group.events.sum()
                be=group[baseline+'_excess'].sum()/group.excess.sum()
                le=group[local+'_excess'].sum()/group.excess.sum()
                return np.array([br,lr,be,le])
            values=contrast(b)
            # Primary uncertainty resamples complete calendar months across all zones.
            # This preserves contemporaneous cross-zone dependence.
            totals=b.groupby('month')[['events','excess',baseline+'_events',local+'_events',baseline+'_excess',local+'_excess']].sum().to_numpy()
            rng=np.random.default_rng(20261004)
            idx=rng.integers(0,len(totals),size=(10000,len(totals)))
            draws=totals[idx].sum(axis=1)
            delta=np.column_stack([(draws[:,3]-draws[:,2])/draws[:,0],(draws[:,5]-draws[:,4])/draws[:,1]])
            ci=np.quantile(delta,[.025,.975],axis=0)
            summary.append({'scope':scope,'comparison':local+' vs '+baseline,'zone_months':len(b),'calendar_months':b.month.nunique(),
                'base_recall':values[0],'local_recall':values[1],'base_excess_share':values[2],'local_excess_share':values[3],
                'delta_recall':values[1]-values[0],'recall_lo':ci[0,0],'recall_hi':ci[1,0],
                'delta_excess_share':values[3]-values[2],'excess_lo':ci[0,1],'excess_hi':ci[1,1]})
            for month in b.month.unique():
                other=b[b.month!=month]
                v=contrast(other)
                audit.append({'scope':scope,'comparison':local+' vs '+baseline,'omitted_month':month,'delta_recall':v[1]-v[0],'delta_excess_share':v[3]-v[2]})
    metrics=[]
    for method in methods:
        g=df[confirm]
        metrics.append({'method':method,**coverage(g.lower_tail_event,g.negative_excess,alerts[method].loc[g.index])})
    pd.DataFrame(metrics).to_csv(out/'nyiso_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'nyiso_paired_calendar_month_bootstrap.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'nyiso_leave_one_month_out.csv',index=False)
    blocks.to_csv(out/'nyiso_zone_month_totals.csv',index=False)
    pd.DataFrame({'timestamp':df.loc[confirm,'timestamp'],'zone':df.loc[confirm,'zone'],'dst_occurrence':df.loc[confirm,'dst_occurrence'],**{k:v.loc[confirm].to_numpy() for k,v in alerts.items()}}).to_csv(out/'nyiso_alerts.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/nyiso/processed/nyiso_clean_dataset.csv'))
    p.add_argument('--out',type=Path,default=Path('results/nyiso'))
    p.add_argument('--historical-only',action='store_true',help='Reconstruct the independent public historical branch')
    a=p.parse_args()
    main(a.data,a.out,a.historical_only)
