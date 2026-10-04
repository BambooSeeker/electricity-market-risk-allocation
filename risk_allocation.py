"""Deterministic cohort allocation and evaluation used by the release demos.

The fixed-count exchange matches the archived Zhejiang implementation.
The optional score frontier is a mathematical diagnostic, not that policy.
"""
from itertools import combinations
import math
import numpy as np
import pandas as pd


def order(values, descending=True):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Scores must be finite')
    return np.argsort(-values if descending else values, kind='stable')


def topk(score, budget):
    if not 0 <= budget <= 1:
        raise ValueError('Budget must be between zero and one')
    out = np.zeros(len(score), dtype=bool)
    out[order(score)[:math.ceil(budget * len(score))]] = True
    return out


def boundary(base, fill, budget=.30, core_margin=.075, pool=.40):
    n = len(base)
    if not 0 <= core_margin <= budget <= pool <= 1:
        raise ValueError('Require 0 <= core_margin <= budget <= pool <= 1')
    k = math.ceil(budget*n)
    kc = math.floor((budget-core_margin)*n)
    idx = order(base)
    core = idx[:kc]
    eligible = idx[kc:math.floor(pool*n)]
    if len(eligible) < k-kc:
        raise ValueError('Candidate pool is too small for the declared capacity')
    chosen = eligible[order(np.asarray(fill)[eligible])[:k-kc]]
    mask = np.zeros(n,dtype=bool)
    mask[np.r_[core,chosen]] = True
    return mask,core


def fixed_count_exchange(base, addition, start, core, stress, pool=.50, cap_fraction=.05):
    n = len(base)
    mask = np.asarray(start,dtype=bool).copy()
    protected = np.zeros(n,dtype=bool)
    protected[core] = True
    rank = np.empty(n,dtype=int)
    rank[order(base)] = np.arange(1,n+1)
    removable = np.flatnonzero(mask & ~protected)
    eligible = np.flatnonzero(~mask & np.asarray(stress,dtype=bool) & (rank/n<=pool))
    cap = math.floor(cap_fraction*mask.sum())
    count = min(cap,len(removable),len(eligible))
    remove = removable[order(np.asarray(base)[removable],False)[:count]]
    add = eligible[order(np.asarray(addition)[eligible])[:count]]
    mask[remove] = False
    mask[add] = True
    assert mask.sum()==np.asarray(start).sum() and mask[core].all()
    return mask, {'count':count,'cap':cap,'removed':remove.tolist(),'added':add.tolist()}


def exchange_frontier(removal_scores,addition_scores,cap):
    removal=np.sort(np.asarray(removal_scores,dtype=float))
    addition=np.sort(np.asarray(addition_scores,dtype=float))[::-1]
    cap=min(cap,len(removal),len(addition))
    gain=np.r_[0.,np.cumsum(addition[:cap]-removal[:cap])]
    efficient=[]
    for count,value in enumerate(gain):
        if count==0 or value>max(gain[:count]):
            efficient.append(count)
    return gain,efficient


def coverage(events,excess,alerts):
    events=np.asarray(events,dtype=float)
    excess=np.asarray(excess,dtype=float)
    alerts=np.asarray(alerts,dtype=bool)
    return {'intervals':len(events),'events':int(events.sum()),'alerts':int(alerts.sum()),
            'recall':float(events[alerts].sum()/events.sum()) if events.sum() else None,
            'excess_share':float(excess[alerts].sum()/excess.sum()) if excess.sum() else None,
            'captured_excess':float(excess[alerts].sum())}


def frozen_ecdf(values,reference):
    ref=np.sort(np.asarray(reference,dtype=float))
    ref=ref[np.isfinite(ref)]
    if not len(ref):
        raise ValueError('Empty calibration reference')
    values=np.asarray(values,dtype=float)
    out=(np.searchsorted(ref,values,side='left')+np.searchsorted(ref,values,side='right'))/(2*len(ref))
    return np.where(np.isfinite(values),out,.5)


def value_frontier(budgets,counts,excess,interval_hours=.5):
    budgets=np.asarray(budgets,dtype=float)
    counts=np.asarray(counts,dtype=float)
    excess=np.asarray(excess,dtype=float)
    if len(budgets)!=len(counts) or len(counts)!=len(excess) or np.any(np.diff(counts)<=0):
        raise ValueError('Ordered capacities must have strictly increasing counts')
    return pd.DataFrame({'budget':budgets[1:],'alerts':counts[1:].astype(int),'captured_excess':excess[1:],
        'normalized_break_even_cost':interval_hours*np.diff(excess)/np.diff(counts)})
