from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True,help='Authorized full-confirmation alert audit CSV')
parser.add_argument('--out',type=Path,default=ROOT/'results/figures')
args=parser.parse_args()
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(args.input,parse_dates=['time'])
assert len(d)==7873
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
COL={'start':'#277DA8','full':'#D5A62D','gate':'#7E9854'}

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',pad_inches=.07)
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight',pad_inches=.07)
    fig.savefig(OUT/f'{stem}.png',dpi=600,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

# Daily evaluation of complete trailing 30-day windows.
d['day']=d.time.dt.floor('D')
daily=d.groupby('day').negative_excess.sum().reindex(pd.date_range(d.day.min(),d.day.max(),freq='D'))
assert daily.notna().all()
den=daily.rolling(30,min_periods=30).sum()
assert den.notna().sum()==136 and (den.dropna()>0).all()
fig,ax=plt.subplots(figsize=(7.2,3.1))
for method,label,ls in [('start','Starting set','--'),('gate','q60 gate','-.'),('full','Full QMLP','-')]:
    captured=(d.negative_excess*d[method+'_alert']).groupby(d.day).sum().reindex(daily.index)
    rate=100*captured.rolling(30,min_periods=30).sum()/den
    ax.plot(rate.index,rate,color=COL[method],lw={'full':1.9,'start':1.5,'gate':1.15}[method],ls=ls,label=label)
ax.set_ylim(0,105)
ax.set_ylabel('30-day ExcessShare (%)')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.grid(axis='y',color='#E6E6E6',lw=.6)
ax.legend(loc='lower left',ncol=3)
save(fig,'fig_05_monthly')

# Pairs follow the exact score-sorted removal and addition lists in the algorithm.
pairs=[]
frontiers=[]
for month,g in d.groupby('test_month'):
    removed=g[g.start_alert & ~g.gate_alert].sort_values('base_score')
    added=g[g.gate_alert & ~g.start_alert].sort_values('c22_deep_w20',ascending=False)
    assert len(removed)==len(added)
    for (_,a),(_,b) in zip(removed.iterrows(),added.iterrows()):
        pairs.append((month,a.base_score,b.c22_deep_w20,b.negative_excess-a.negative_excess))
    if len(removed):
        gain=added.c22_deep_w20.to_numpy()-removed.base_score.to_numpy()
        effect=added.negative_excess.to_numpy()-removed.negative_excess.to_numpy()
        frontiers.append((month,np.cumsum(gain),effect))
assert len(pairs)==65 and all(b>a for _,a,b,_ in pairs)
fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained')
for (month,gain,effect),color,style in zip(frontiers,
        ['#D5A62D','#A97B18','#7E9854'],['-','--','-.']):
    change=2*np.arange(1,len(gain)+1)
    ax.plot(np.r_[0,change],np.r_[0,gain],color=color,lw=1.65,ls=style,
            label=pd.Timestamp(month).strftime('%b %Y'))
    ax.scatter(change,gain,s=13,facecolors='white',edgecolors=color,lw=.65,zorder=3)
    for mask,c,marker in [(effect>0,COL['full'],'^'),(effect<0,COL['start'],'v')]:
        ax.scatter(change[mask],gain[mask],s=44,color=c,marker=marker,
                   edgecolors='#333333',linewidths=.45,zorder=4)
ax.set_xlim(-1,46)
ax.set_ylim(bottom=0)
ax.set_xticks([0,10,20,30,40,44])
ax.set_xlabel(r'Changed memberships, $2\ell$')
ax.set_ylabel(r'Cumulative priority gain, $F_m(\ell)$')
ax.grid(color='#E6E6E6',lw=.5)
ax.set_axisbelow(True)
legend=ax.legend(loc='upper left',ncol=3,fontsize=7.5)
ax.add_artist(legend)
ax.legend(handles=[Line2D([],[],ls='',marker='^',color=COL['full'],label='Excess gained'),
                   Line2D([],[],ls='',marker='v',color=COL['start'],label='Excess lost')],
          loc='lower right',ncol=2,fontsize=7.5)
save(fig,'fig_06_exchanges')
print('All 7873 records used; 136 complete 30-day windows; all 65 algorithm-ordered exchanges displayed.')
