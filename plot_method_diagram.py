from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'results/figures'
FIG.mkdir(parents=True,exist_ok=True)
AUD=ROOT/'data/summary'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#285c79','#a33d4c','#45816b']


# New diagram is generated from the mathematical selection specification.
fig,axes=plt.subplots(2,1,figsize=(8,4.5),layout='constrained')
for ax in axes:
    ax.set_xlim(-.5,15.5);ax.set_ylim(-.5,2.2);ax.axis('off')
for i in range(16):
    color=COLORS[1] if i<6 else '#d19b45' if i<12 else '#d5d5d5'
    axes[0].add_patch(Rectangle((i,.3),.85,.65,facecolor=color,edgecolor='black',lw=.6))
axes[0].text(0,1.65,'a  Core-preserving boundary selection',fontweight='bold')
axes[0].text(2.5,1.12,r'Protected core $C_m$',ha='center')
axes[0].text(8.5,1.12,r'Candidate pool $U_m$',ha='center')
axes[0].text(13.5,1.12,'Remainder',ha='center')
axes[0].text(7.5,-.15,r'Fill remaining $K_m-|C_m|$ places by $r^b$',ha='center')
axes[1].text(0,1.65,'b  Stress-conditioned exchange',fontweight='bold')
for i in range(12):
    color=COLORS[1] if i<6 else '#d19b45' if i<9 else COLORS[2]
    axes[1].add_patch(Rectangle((i,.3),.85,.65,facecolor=color,edgecolor='black',lw=.6))
axes[1].text(2.5,1.12,'Core retained',ha='center')
axes[1].text(8.3,-.20,r'Lowest $r^0$ removed; highest eligible $r^u$ added',ha='center')
axes[1].text(10.1,1.12,r'$g_t=1$',ha='center')
axes[1].text(13.2,.60,r'$\ell=\min(L_m,|R_m|,|S_m|)$',ha='center',fontsize=9)
fig.savefig(FIG/'fig_02_method.pdf',bbox_inches='tight');plt.close(fig)

