from pathlib import Path
import pandas as pd
from risk_allocation import coverage


def main():
    root=Path(__file__).resolve().parent
    frame=pd.read_csv(root/'data/zhejiang_demo/week_saved_score_audit.csv')
    assert len(frame)==336
    results=[]
    for name in ['start','full','gate']:
        results.append({'method':name,**coverage(frame.negative_tail,frame.negative_excess,frame[name+'_alert'])})
    removed=frame.start_alert & ~frame.gate_alert
    added=frame.gate_alert & ~frame.start_alert
    # Week slices need not balance: equal cardinality is a whole-month invariant.
    out=root/'results/saved_week';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame([{'removed_in_week':int(removed.sum()),'added_in_week':int(added.sum())}]).to_csv(out/'exchange_slice.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__=='__main__':main()
