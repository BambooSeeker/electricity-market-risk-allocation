"""Rebuild the original four profile panels with the shared results palette."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--monthly',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.audit,parse_dates=['time'])
m=pd.read_csv(a.monthly)
assert len(d)==7873 and d.negative_tail.sum()==624
assert (m.events.to_numpy()==d.groupby('test_month').negative_tail.sum().to_numpy()).all()
d['hour']=d.time.dt.hour
h=d.groupby(['test_month','hour']).negative_tail.mean().unstack(0)*100
yellow,blue,green,gold='#D5A62D','#277DA8','#7E9854','#A97B18'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':7,'legend.frameon':False,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
x=np.arange(6)
labels=[pd.Timestamp(s).strftime('%b %y') for s in m.month]
ax=axs[0,0]
ax.bar(x,100*m.events/m.intervals,color=[yellow]*2+[blue]*4,width=.65)
ax.set_ylabel('Tail-event rate (%)')
ax.set_xticks(x,labels,rotation=30)
ax=axs[0,1]
ax.plot(x,m.abs_spread_q95,'o-',color=blue,ms=3,label=r'$|s|$ q95')
ax.plot(x,m.spread_sd,'s--',color=yellow,ms=3,label='Spread SD')
ax.set_ylabel('Spread (CNY/MWh)')
ax.set_xticks(x,labels,rotation=30)
ax.legend(loc='upper right')
ax=axs[1,0]
for month,color,ls in [('2025-09',yellow,'-'),('2025-10',gold,'-'),
                        ('2025-11',green,'--'),('2026-02',blue,'--')]:
    ax.plot(h.index,h[month],color=color,ls=ls,lw=1.5,marker='o',ms=2,
            label=pd.Timestamp(month).strftime('%b %y'))
ax.set_xlabel('Hour of day')
ax.set_ylabel('Tail-event rate (%)')
ax.set_xticks([0,6,12,18,23])
ax.set_ylim(-1,54)
ax.legend(loc='upper right',ncol=2)
ax=axs[1,1]
groups=[m.iloc[:2],m.iloc[2:]]
rate=[100*(g.events/g.intervals).mean() for g in groups]
q95=[g.abs_spread_q95.mean() for g in groups]
ax.bar(np.arange(2)-.18,rate,width=.35,color=yellow,label='Event rate')
ax2=ax.twinx()
ax2.bar(np.arange(2)+.18,q95,width=.35,color=blue,label=r'$|s|$ q95')
ax2.spines['right'].set_visible(True)
ax.set_ylabel('Tail-event rate (%)')
ax2.set_ylabel(r'$|s|$ q95 (CNY/MWh)')
ax.set_xticks([0,1],['Pressure','Other months'])
ax.legend(handles=ax.patches[:1]+ax2.patches[:1],labels=['Event rate',r'$|s|$ q95'],
          loc='upper right')
for ax,letter in zip(axs.flat,'abcd'):
    ax.text(-.16,1.03,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
fig.savefig(a.out/'fig_03_profile.pdf',bbox_inches='tight',pad_inches=.07)
fig.savefig(a.out/'fig_03_profile.svg',bbox_inches='tight',pad_inches=.07)
fig.savefig(a.out/'fig_03_profile.png',dpi=600,bbox_inches='tight',pad_inches=.07)
plt.close(fig)
print('Original four profile comparisons retained; palette and typography harmonized.')
