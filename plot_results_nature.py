"""Reproduce Figures 3--7 from disclosed aggregate evidence. BSD-3-Clause."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/summary'
OUT=ROOT/'results/figures'
OUT.mkdir(parents=True,exist_ok=True)
COL={'starting':'#909995','hgb':'#909995','full':'#D99C9E','core':'#E5CF87',
     'gate':'#9CBFA6','pressure':'#D99C9E','other':'#909995',
     'recall':'#9CBFA6','excess':'#D99C9E','removed':'#D99C9E','added':'#9CBFA6'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,
 'axes.edgecolor':'#555555','text.color':'#252525','axes.labelcolor':'#252525'})

def title(ax,letter,label):
    ax.set_title(label,loc='left',pad=10)
    ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')

def grid(ax,axis='y'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color='#E6E8EA',linewidth=.6)

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{stem}.png',dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def pressure_profile():
    m=pd.read_csv(DATA/'zhejiang_operating_profile.csv')
    p=pd.read_csv(DATA/'zhejiang_intraday_profile.csv')
    s=pd.read_csv(DATA/'zhejiang_pressure_concentration.csv').set_index('stratum')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained')
    x=np.arange(len(m));labels=m.month.str.slice(2)
    ax=axs[0,0]
    rate=100*m.events/m.intervals
    ax.bar(x,rate,color=[COL['pressure']]*2+[COL['other']]*4,width=.65)
    for i,(r,n) in enumerate(zip(rate,m.events)):
        ax.text(i,r+.5,f'{r:.1f}%',ha='center',fontsize=8)
    ax.set_xticks(x,labels,rotation=30);ax.set_ylim(0,24);ax.set_ylabel('Tail-event rate (%)')
    title(ax,'a','Monthly event concentration');grid(ax)
    ax=axs[0,1]
    ax.plot(x,m.spread_sd,'o-',color=COL['starting'],label='Spread SD',ms=4)
    ax.plot(x,m.abs_spread_q95,'D--',color=COL['pressure'],label=r'$|s|$ 95th percentile',ms=4)
    ax.set_xticks(x,labels,rotation=30);ax.set_ylabel('Spread magnitude (CNY/MWh)');ax.set_ylim(0,440)
    ax.legend(frameon=False,loc='upper center',ncol=2,fontsize=7);title(ax,'b','Monthly spread dispersion');grid(ax)
    ax=axs[1,0]
    for name,c,ls in [('Pressure',COL['pressure'],'-'),('Other months',COL['other'],'--')]:
        v=p[p.stratum.eq(name)]
        ax.plot(v.hour,100*v.event_rate,color=c,ls=ls,lw=1.6,label=name)
    ax.set_xlim(0,23.5);ax.set_xticks([0,6,12,18,23]);ax.set_ylim(bottom=0)
    ax.set_xlabel('Hour of day');ax.set_ylabel('Tail-event rate (%)');ax.legend(frameon=False,loc='upper left')
    title(ax,'c','Intraday risk profile');grid(ax)
    ax=axs[1,1]
    vals=100*s.loc['Pressure',['intervals','events','excess']].to_numpy(float)/s[['intervals','events','excess']].sum().to_numpy(float)
    y=np.arange(3)
    ax.barh(y,vals,color=COL['pressure'],height=.52,label='Pressure')
    ax.barh(y,100-vals,left=vals,color=COL['other'],height=.52,label='Other months')
    for i,v in enumerate(vals):ax.text(v/2,i,f'{v:.1f}%',ha='center',va='center',color='white',fontweight='bold')
    ax.set_yticks(y,['Intervals','Tail events','Threshold excess']);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Share of confirmation total (%)');title(ax,'d','Pressure-period concentration')
    save(fig,'fig_03_profile')

def coverage():
    d=pd.read_csv(DATA/'zhejiang_coverage_comparison.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True,layout='constrained')
    for ax,metric,heading,letter in zip(axs,['recall','excess_share'],['Event coverage','Severity coverage'],'ab'):
        for i,r in d.iterrows():
            lo=100*r['pressure_'+metric];hi=100*r['all_'+metric];c=COL[r.method]
            ax.plot([lo,hi],[i,i],color=c,lw=1.5,alpha=.65)
            ax.scatter([hi],[i],color=c,s=40,zorder=3)
            ax.scatter([lo],[i],edgecolor=c,facecolor='white',marker='D',s=35,zorder=3)
            ax.text(hi+1.1,i-.12,f'{hi:.2f}',color=c,fontsize=8)
            ax.text(lo-1.1,i+.27,f'{lo:.2f}',ha='right',color=c,fontsize=8)
        ax.axvline(30,color='#858585',ls=':',lw=1)
        ax.set_xlim(25,99);ax.set_ylim(3.65,-.65);ax.set_xlabel(('Recall' if metric=='recall' else 'ExcessShare')+' (%)')
        ax.set_yticks(np.arange(4),d.label);grid(ax,'x');title(ax,letter,heading)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#555',label='All months'),
        Line2D([],[],marker='D',ls='',mfc='white',color='#555',label='Pressure months'),
        Line2D([],[],ls=':',color='#858585',label='Uniform allocation (~30%)')],loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,fontsize=8,frameon=False)
    save(fig,'fig_04_coverage')

def monthly():
    d=pd.read_csv(DATA/'zhejiang_chronological_monthly.csv');months=sorted(d.month.unique());x=np.arange(len(months))
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
    for j,metric in enumerate(['recall','excess_share']):
        for method,label,mark,ls in [('starting','Starting set','s','--'),('full','Full QMLP','o','-'),('gate','Gate q60','D','-')]:
            v=d[d.method.eq(method)].set_index('month').loc[months,metric]*100
            axs[0,j].plot(x,v,marker=mark,ms=4,ls=ls,lw=1.3,color=COL[method],label=label)
        base=d[d.method.eq('starting')].set_index('month').loc[months,metric]*100
        for method,offset,mark in [('full',-.06,'o'),('gate',.06,'D')]:
            delta=d[d.method.eq(method)].set_index('month').loc[months,metric]*100-base
            axs[1,j].plot(x+offset,delta,marker=mark,ms=4,color=COL[method],lw=1.1)
        for ax in axs[:,j]:
            ax.axvspan(-.45,1.45,color=COL['pressure'],alpha=.07,zorder=0)
            counts=d[d.method.eq('starting')].set_index('month').loc[months,'events']
            ax.set_xticks(x,[f'{s[2:]}\nn={int(n)}' for s,n in zip(months,counts)]);grid(ax)
        axs[0,j].set_ylim(50,103);axs[0,j].set_ylabel('Coverage (%)')
        axs[1,j].axhline(0,color='#777',lw=.8);axs[1,j].set_ylabel('Change (percentage points)')
        axs[1,j].relim();axs[1,j].autoscale_view();axs[1,j].margins(y=.18)
        axs[1,j].set_xlabel('Evaluation month / tail-event count')
    title(axs[0,0],'a','Monthly Recall');title(axs[0,1],'b','Monthly ExcessShare')
    title(axs[1,0],'c','Recall relative to starting set');title(axs[1,1],'d','ExcessShare relative to starting set')
    axs[0,0].legend(frameon=False,loc='upper left',fontsize=7)
    save(fig,'fig_05_monthly')

def exchanges():
    d=pd.read_csv(DATA/'zhejiang_realized_exchange_totals.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True,layout='constrained')
    for ax,metric,letter,heading in zip(axs,['events','excess'],'ab',['Exchanged tail events','Exchanged threshold excess']):
        maxval=max(d['removed_'+metric].max(),d['added_'+metric].max())
        for i,r in d.iterrows():
            a,b=r['removed_'+metric],r['added_'+metric]
            ax.plot([a,b],[i,i],color='#B3B8BC',lw=2,zorder=1)
            ax.scatter(a,i,marker='s',s=42,color=COL['removed'],zorder=3)
            ax.scatter(b,i,marker='o',s=42,color=COL['added'],zorder=3)
            ax.annotate(f'{a:g}' if metric=='events' else f'{a:.1f}',(a,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=COL['removed'])
            ax.annotate(f'{b:g}' if metric=='events' else f'{b:.1f}',(b,i),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=8,color=COL['added'])
        ax.set_xlim(-.08*maxval,1.2*maxval);ax.set_ylim(len(d)-.4,-.6)
        ax.set_yticks(range(len(d)),[f'{r.month[2:]}  (n = {int(r.exchanges)})' for _,r in d.iterrows()])
        ax.set_xlabel('Event count' if metric=='events' else 'Threshold excess (CNY/MWh)');grid(ax,'x');title(ax,letter,heading)
    axs[0].legend(handles=[Line2D([],[],ls='',marker='s',color=COL['removed'],label='Removed'),Line2D([],[],ls='',marker='o',color=COL['added'],label='Added')],frameon=False,loc='lower right',fontsize=7)
    save(fig,'fig_06_exchanges')

def robustness():
    g=pd.read_csv(DATA/'zhejiang_chronological_gate_summary.csv');g=g[g.scope.eq('pressure')]
    ny=pd.read_csv(ROOT/'results/nyiso/nyiso_metrics.csv').set_index('method')
    ci=pd.read_csv(ROOT/'results/nyiso/nyiso_paired_calendar_month_bootstrap.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.5),layout='constrained',gridspec_kw={'height_ratios':[1,1.2]})
    ax=axs[0,0]
    ax.plot(g.recall*100,g.excess_share*100,'D-',color=COL['gate'],ms=5,lw=1.1)
    for _,r in g.iterrows():ax.annotate(f'q{int(r.q*100)}',(r.recall*100,r.excess_share*100),xytext=(6,-13) if r.q==.8 else (6,7),textcoords='offset points',fontsize=8)
    ax.scatter(70.0696056,78.8576,marker='s',color=COL['starting'],s=35)
    ax.annotate('Starting set',(70.0696056,78.8576),xytext=(-28,12),textcoords='offset points',ha='right',fontsize=8,arrowprops={'arrowstyle':'-','color':COL['starting'],'lw':.6})
    ax.set_xlabel('Recall (%)');ax.set_ylabel('ExcessShare (%)');ax.margins(.25);grid(ax);title(ax,'a','Zhejiang gate threshold')
    ax=axs[0,1]
    for a,b,c,label in [('base','local',COL['starting'],'Historical'),('source_base','source_local',COL['full'],'Source')]:
        v=ny.loc[[a,b]]
        ax.plot(v.recall*100,v.excess_share*100,color=c,lw=1.2)
        ax.scatter(v.iloc[0].recall*100,v.iloc[0].excess_share*100,marker='o',facecolors='white',edgecolors=c,s=38)
        ax.scatter(v.iloc[1].recall*100,v.iloc[1].excess_share*100,marker='D',color=c,s=38,label=label)
    ax.set_xlabel('Recall (%)');ax.set_ylabel('ExcessShare (%)');ax.margins(.25)
    ax.legend(handles=[Line2D([],[],color=COL['starting'],label='Historical'),Line2D([],[],color=COL['full'],label='Source'),Line2D([],[],ls='',marker='o',mfc='white',color='#666',label='Base'),Line2D([],[],ls='',marker='D',color='#666',label='Local')],frameon=False,loc='lower left',fontsize=7,ncol=2)
    title(ax,'b','NYISO score origin');grid(ax)
    rows=[]
    for scope,comp,label in [('all','local vs base','All / historical (6)'),('pressure','local vs base','Pressure 3/4 (4)'),('strict_pressure','local vs base','Pressure 4/4 (3)'),('high_load','local vs base','High load (5)'),('all','source_local vs source_base','All / source (6)')]:
        row=ci[(ci.scope.eq(scope)) & (ci.comparison.eq(comp))].iloc[0]
        rows.append((label,row))
    for ax,metric,letter,name in zip(axs[1],['recall','excess'],'cd',['Local adjustment: Recall','Local adjustment: ExcessShare']):
        for i,(label,r) in enumerate(rows):
            value=r['delta_recall' if metric=='recall' else 'delta_excess_share']*100
            lo,hi=r[metric+'_lo']*100,r[metric+'_hi']*100
            c=COL['recall' if metric=='recall' else 'excess']
            ax.plot([lo,hi],[i,i],color=c,lw=1.3);ax.plot([lo,lo],[i-.07,i+.07],color=c,lw=1)
            ax.plot([hi,hi],[i-.07,i+.07],color=c,lw=1);ax.scatter(value,i,color=c,s=24,zorder=3)
        ax.axvline(0,color='#777',ls='--',lw=.8);ax.set_ylim(4.6,-.6)
        ax.set_yticks(range(5),[r[0] for r in rows] if metric=='recall' else ['']*5)
        ax.set_xlabel('Change (percentage points)');grid(ax,'x');title(ax,letter,name)
    save(fig,'fig_07_robustness')

if __name__=='__main__':
    pressure_profile();coverage();monthly();exchanges();robustness()
    (OUT/'palette.json').write_text(json.dumps(COL,indent=2),encoding='utf-8')
    print('Figures 3--7 exported as editable PDF/SVG and 600-dpi PNG.')
