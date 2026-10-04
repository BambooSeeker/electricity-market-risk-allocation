from pathlib import Path
import pandas as pd
from risk_allocation import value_frontier


def main():
    root=Path(__file__).resolve().parent
    # Archived budget-specific policy profile reported in the manuscript.
    profile=pd.DataFrame({'budget':[0,.1,.2,.3,.4],
                          'alerts':[0,788,1576,2364,3152],
                          'excess':[0,31951.9,48833.3,59647.7,63222.9]})
    frontier=value_frontier(profile.budget,profile.alerts,profile.excess,.5)
    profile['value_at_normalized_cost_4']=.5*profile.excess-4*profile.alerts
    best=profile.loc[profile.value_at_normalized_cost_4.idxmax(),'budget']
    assert best==.3
    out=root/'results/capacity_value';out.mkdir(parents=True,exist_ok=True)
    profile.to_csv(out/'policy_profile.csv',index=False)
    frontier.to_csv(out/'break_even_costs.csv',index=False)
    print(frontier.to_string(index=False));print('Optimal candidate budget at cost 4:',best)


if __name__=='__main__':main()
