"""Single-axis results figures from archived aggregate evidence. BSD-3-Clause."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/summary'
OUT = ROOT/'results/figures'
OUT.mkdir(parents=True, exist_ok=True)
COL = {'starting':'#277DA8','full':'#D5A62D','gate':'#7E9854','core':'#A97B18'}
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def save(fig, stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',pad_inches=.07)
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight',pad_inches=.07)
    fig.savefig(OUT/f'{stem}.png',dpi=600,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

def monthly():
    d = pd.read_csv(DATA/'zhejiang_chronological_monthly.csv')
    months = sorted(d.month.unique())
    fig,ax=plt.subplots(figsize=(6.4,3.1))
    for name,label,marker,style in [('starting','Starting set','s','--'),
        ('full','Full QMLP','o','-'),('gate','q60 gate','D','-')]:
        v=d[d.method.eq(name)].set_index('month').loc[months]
        ax.plot(range(6),100*v.excess_share,color=COL[name],marker=marker,
                linestyle=style,lw=1.7,ms=5,markeredgecolor='#555555',markeredgewidth=.45,label=label)
    ax.set_xticks(range(6),[pd.Timestamp(m).strftime('%b %Y') for m in months])
    ax.set_ylabel('ExcessShare (%)')
    ax.set_ylim(0,105)
    ax.grid(axis='y',color='#E7E7E7',lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left',ncol=3)
    save(fig,'fig_05_monthly')

def exchanges():
    d=pd.read_csv(DATA/'zhejiang_realized_exchange_totals.csv')
    fig,ax=plt.subplots(figsize=(5.5,3.0))
    x=np.arange(len(d))
    ax.bar(x-.18,d.removed_excess,width=.34,color=COL['starting'],edgecolor='#6F370F',lw=.45,label='Removed')
    ax.bar(x+.18,d.added_excess,width=.34,color=COL['gate'],edgecolor='#395B1C',lw=.45,label='Added')
    ax.set_xticks(x,[pd.Timestamp(m).strftime('%b %Y') for m in d.month])
    ax.set_ylabel('Threshold-excess sum (CNY/MWh)')
    ax.set_ylim(0,320)
    ax.grid(axis='y',color='#E7E7E7',lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right',ncol=2)
    save(fig,'fig_06_exchanges')

def regional_effects():
    d=pd.read_csv(ROOT/'results/nyiso/nyiso_zone_month_totals.csv')
    assert len(d)==66 and d.zone.nunique()==11
    months=sorted(d.month.unique())
    zones=['WEST','GENESE','CENTRL','NORTH','MHK VL','CAPITL','HUD VL','MILLWD','DUNWOD','N.Y.C.','LONGIL']
    if set(zones)!=set(d.zone):
        zones=sorted(d.zone.unique())
    arrays=[]
    for base,local in [('base','local'),('source_base','source_local')]:
        block=d.assign(effect=100*(d[local+'_excess']-d[base+'_excess'])/d.excess.replace(0,np.nan))
        arrays.append(block.pivot(index='zone',columns='month',values='effect').reindex(index=zones,columns=months).to_numpy())
    values=np.concatenate(arrays,axis=1)
    limit=max(5,float(np.nanmax(np.abs(values))))
    cmap=LinearSegmentedColormap.from_list('local_effect',['#185B88','#62A0C3','#C3B669','#E1C358','#A77B16'])
    cmap.set_bad('#E8E8E8')
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    im=ax.imshow(values,aspect='auto',interpolation='nearest',cmap=cmap,
        norm=SymLogNorm(linthresh=1,linscale=.5,vmin=-limit,vmax=limit,base=10))
    ax.set_yticks(range(11),zones)
    ax.set_xticks(range(12),[pd.Timestamp(m).strftime('%b') for m in months]*2)
    ax.set_xticks(np.arange(-.5,12,1),minor=True)
    ax.set_yticks(np.arange(-.5,11,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.75)
    ax.tick_params(which='minor',length=0)
    ax.axvline(5.5,color='#555555',lw=1.2)
    ax.text(2.5,-1.05,'Historical score',ha='center',fontsize=9)
    ax.text(8.5,-1.05,'Zhejiang-source score',ha='center',fontsize=9)
    ax.set_xlabel('Evaluation month, 2025')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.19,fraction=.045,aspect=35,
        ticks=[-60,-20,-5,-1,0,1,5,20,60])
    cb.set_ticklabels(['-60','-20','-5','-1','0','1','5','20','60'])
    cb.set_label('Change in ExcessShare (percentage points)\nSymmetric log scale',fontsize=8)
    save(fig,'fig_07_robustness')
    pd.DataFrame(values,index=zones,columns=['historical_'+m for m in months]+['source_'+m for m in months]).to_csv(OUT/'fig_07_source.csv')

if __name__=='__main__':
    regional_effects()
    (OUT/'palette.json').write_text(json.dumps(COL,indent=2),encoding='utf-8')
    print('Figure 7 exported; all 66 NYISO zone-months retained. Figures 4--6 require authorized Zhejiang history.')
