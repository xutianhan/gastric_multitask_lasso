#!/usr/bin/env python3
"""Render the expanded framework and audited EAC/STAD comparisons."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle

from gastric_mtl.paths import EAC_STAD, FIGURES
RUN=EAC_STAD
OUT=FIGURES/'eac_stad'
COLORS=['#4d7188','#b16e5f'];MARKERS=['o','s']


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=400)
    plt.close(fig)


def framework():
    # Restore the earlier training/test branch layout. No observed counts,
    # fitted feature counts, effect estimates or conclusions belong here.
    fig=plt.figure(figsize=(7,5.8),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7),ylim=(0,5.8));ax.axis('off')
    ink='#243d4e';blue='#356685';edge='#a4b9c7';fill='#f4f7f9'
    def txt(x,y,s,size=9.5,weight='normal',color=ink,ha='center'):
        return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va='center',color=color,linespacing=1.35)
    def box(x,y,w,h,fc=fill,ec=edge,lw=.8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.055',facecolor=fc,edgecolor=ec,linewidth=lw))
    def arrow(x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=blue,lw=1.0))
    box(.06,5.12,6.88,.6)
    txt(3.5,5.53,'1  TCGA cohorts and aligned inputs',10.5,'bold')
    txt(3.5,5.29,'ESCA · STAD · COAD    |    RNA sequencing · overall survival · clinical data',9)
    arrow(3.5,5.10,3.5,4.94)
    box(.06,4.43,6.88,.49)
    txt(3.5,4.76,'2  Nested cross-validation',10.5,'bold')
    txt(3.5,4.55,'Five outer folds × two partitions',9.5)
    arrow(2.61,4.41,2.61,4.23);arrow(6.14,4.41,6.14,4.23)
    box(.06,.89,5.10,3.31,fc='white',ec=blue,lw=1.1)
    txt(2.61,4.01,'3  Outer training set',10.5,'bold')
    box(.20,3.26,4.82,.52)
    txt(2.61,3.60,'Training-only preprocessing',9.7,'bold')
    txt(2.61,3.39,'Log transform → variance filter → scaling → pathway groups',8.9)
    arrow(2.61,3.24,2.61,3.10)
    box(.20,1.82,4.82,1.26)
    txt(2.61,2.89,'Candidate sharing structures',9.7,'bold')
    graphs=[[],[(0,1)],[(0,2)],[(1,2)],[(0,1),(0,2),(1,2)]]
    for j,(label,edges) in enumerate(zip(['Single','E + S','E + C','S + C','Three tasks'],graphs)):
        x=.67+.97*j;coords=[(x-.18,2.39),(x+.18,2.39),(x,2.08)]
        for a,b in edges:ax.plot([coords[a][0],coords[b][0]],[coords[a][1],coords[b][1]],color=blue,lw=1.3,zorder=1)
        active={v for e in edges for v in e} if edges else {0,1,2}
        for k,((cx,cy),letter) in enumerate(zip(coords,['E','S','C'])):
            color=blue if k in active else '#adbac3'
            ax.add_patch(Circle((cx,cy),.106,fc='white',ec=color,lw=1.05,zorder=2))
            txt(cx,cy,letter,8.2,color=color)
        txt(x,2.66,label,8.7)
    arrow(2.61,1.80,2.61,1.66)
    box(.20,.99,4.82,.65)
    txt(2.61,1.46,'Cox loss + smooth Group Lasso + directional coupling',9.1,'bold')
    txt(2.61,1.15,'Inner 3-fold selection of λ, μ and graph → refit model',9.0)
    # Hold-out data remain separate until the final prediction step.
    box(5.36,.89,1.58,3.31,fc='#edf3f7')
    txt(6.15,3.91,'Outer test fold',9.8,'bold')
    txt(6.15,3.09,'Held-out\npatients',9.8)
    arrow(6.15,2.76,6.15,2.39)
    txt(6.15,2.01,'Apply fitted\ntransforms\nand model',9.4)
    txt(6.15,1.23,'Predict survival',9.2)
    arrow(5.18,1.20,5.35,1.20)
    arrow(2.61,.87,2.61,.70);arrow(6.15,.87,6.15,.70)
    box(.06,.07,6.88,.61)
    txt(3.5,.49,'4  Held-out evaluation',10.5,'bold')
    txt(3.5,.25,'C-index · 1- and 3-year IPCW Brier · clinical reference and augmentation',9.2)
    save(fig,'Figure1_method_framework')


def results():
    facts=json.loads((RUN/'eac_stad_facts.json').read_text())
    assert facts['complete'] and facts['validation']['passed']
    R=facts['results'];metrics=['c_index','brier_1y','brier_3y']
    titles=['C-index difference','1-year Brier difference','3-year Brier difference']
    fig,axes=plt.subplots(1,3,figsize=(7,2.45),sharey=True)
    compact_titles=['ΔC-index','ΔBrier at 1 year','ΔBrier at 3 years']
    for ki,(ax,key,title) in enumerate(zip(axes,metrics,titles)):
        for ci,c in enumerate(['EAC','STAD']):
            for si,seed in enumerate(['42','2026']):
                v=R[seed][c]['pair_EAC_STAD'][key+'_vs_single'];x=v['difference'];lo,hi=v['ci95'];y=1-ci+(.13 if si==0 else -.13)
                ax.hlines(y,lo,hi,color=COLORS[si],lw=1.15)
                ax.vlines([lo,hi],y-.045,y+.045,color=COLORS[si],lw=1)
                ax.plot(x,y,marker=MARKERS[si],color=COLORS[si],markersize=4.5,ls='',label=('Seed '+seed) if ci==0 else None)
        ax.axvline(0,color='#787878',lw=.8,ls='--');ax.grid(axis='x',alpha=.17)
        ax.set_yticks([0,1]);ax.set_yticklabels(['STAD','EAC']);ax.set_ylim(-.45,1.45)
        ax.set_title(chr(97+ki)+'  '+compact_titles[ki],loc='left',fontsize=10,fontweight='bold')
        ax.set_xlabel('Higher favours sharing' if key=='c_index' else 'Lower favours sharing',fontsize=8.5)
        ax.tick_params(labelsize=9)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.55,.975))
    fig.subplots_adjust(left=.095,right=.985,top=.76,bottom=.24,wspace=.29)
    save(fig,'Figure3_EAC_STAD_contrasts')
    fig,axes=plt.subplots(2,3,figsize=(7,5.0),sharex=True)
    for ci,c in enumerate(['EAC','STAD']):
        for ki,(key,title) in enumerate(zip(metrics,titles)):
            ax=axes[ci,ki]
            for si,seed in enumerate([42,2026]):
                v=sorted([v for v in facts['strength_curves'] if v['seed']==seed and v['cancer']==c],key=lambda x:x['mu'])
                ax.plot([x['mu'] for x in v],[x[key+'_vs_single']['difference'] for x in v],color=COLORS[si],marker=MARKERS[si],markersize=3.5,lw=1.1,label='Seed '+str(seed))
            ax.axhline(0,color='#787878',lw=.7,ls='--');ax.set_xscale('log');ax.grid(alpha=.15)
            ax.set_title(c+' · '+title,fontsize=9);ax.tick_params(labelsize=8)
            if ci==1:ax.set_xlabel('Coupling μ',fontsize=9)
            if ki==0:ax.set_ylabel('Difference from single task',fontsize=8)
    handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.10,right=.985,top=.93,bottom=.15,wspace=.42,hspace=.40)
    save(fig,'Figure4_EAC_STAD_strength_response')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--framework-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    for path in (FIGURES/'three_cancer').glob('*'):
        if path.name.startswith(('Figure2_','FigureS1_')):shutil.copy2(path,OUT/path.name)
    framework()
    if not args.framework_only:results()
    print(OUT)

if __name__=='__main__':main()
