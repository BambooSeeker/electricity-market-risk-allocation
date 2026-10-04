"""A chronological one-week HGB demonstration with compact disclosed features."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from risk_allocation import boundary, coverage, fixed_count_exchange, topk


def main(data,out):
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data,parse_dates=['time']).sort_values('time').reset_index(drop=True)
    if len(df)!=336 or df.time.diff().dropna().ne(pd.Timedelta(minutes=30)).any():
        raise ValueError('Demo requires exactly one contiguous half-hourly week')
    train=df.iloc[:192].copy(); cal=df.iloc[192:240].copy(); test=df.iloc[240:].copy()
    tau=float(train.spread_rt_minus_da.quantile(.1))
    features=['da_price_mean','load_day_ahead_pred','net_load_pred','da_cong_node_abs_mean','hour_sin','hour_cos']
    for g in [train,cal,test]:
        hour=g.time.dt.hour+g.time.dt.minute/60
        g['hour_sin']=np.sin(2*np.pi*hour/24); g['hour_cos']=np.cos(2*np.pi*hour/24)
    x=train[features]; y=train.spread_rt_minus_da.le(tau).astype(int)
    clf=HistGradientBoostingClassifier(learning_rate=.035,max_iter=220,max_leaf_nodes=31,early_stopping=False,random_state=20261004).fit(x,y)
    severity=np.maximum(tau-train.spread_rt_minus_da,0.)
    reg=HistGradientBoostingRegressor(learning_rate=.045,max_iter=260,max_leaf_nodes=31,early_stopping=False,random_state=20261004).fit(x,severity)
    pcal=clf.predict_proba(cal[features])[:,1]
    iso=IsotonicRegression(out_of_bounds='clip').fit(pcal,cal.spread_rt_minus_da.le(tau).astype(int))
    probability=iso.predict(clf.predict_proba(test[features])[:,1])
    severity=np.maximum(reg.predict(test[features]),0.)
    # Ranking is explicitly a retrospective capacity benchmark over the demo test cohort.
    score=.6*pd.Series(probability).rank(pct=True)+.4*pd.Series(severity).rank(pct=True)
    local=pd.Series(test.da_cong_node_abs_mean.to_numpy()).rank(pct=True)
    fill=score+.2*local
    coremask,core=boundary(score.to_numpy(),fill.to_numpy(),.3,.075,.4)
    gate_threshold=float(cal.net_load_pred.quantile(.6))
    stress=(test.net_load_pred>=gate_threshold).to_numpy()
    gated,exchange=fixed_count_exchange(score.to_numpy(),fill.to_numpy(),coremask,core,stress,.5,.05)
    events=test.spread_rt_minus_da.le(tau).to_numpy()
    excess=np.maximum(tau-test.spread_rt_minus_da.to_numpy(),0.)
    metrics=[]
    for method,alerts in [('base',topk(score,.3)),('core_preserved',coremask),('stress_gate',gated)]:
        metrics.append({'method':method,**coverage(events,excess,alerts)})
    test=test.assign(tail_event=events,negative_excess=excess,probability=probability,severity=severity,base_score=score.to_numpy(),core_alert=coremask,gate_alert=gated)
    test.to_csv(out/'demo_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'demo_metrics.csv',index=False)
    settings={'seed':20261004,'train_rows':192,'calibration_rows':48,'test_rows':96,'tail_threshold':tau,
              'gate_threshold':gate_threshold,'features':features,'exchange':exchange,
              'scope':'Compact HGB demonstration; paper ensembles and full-period metrics require the authorized full history.'}
    (out/'demo_run.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('results/demo'))
    a=p.parse_args(); main(a.data,a.out)
