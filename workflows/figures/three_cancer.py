#!/usr/bin/env python3
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gastric_mtl.paths import THREE_CANCER, FIGURES
RUN=THREE_CANCER
OUT=FIGURES/'three_cancer'

def main():
 facts=json.loads((RUN/'rerun_facts.json').read_text());assert facts['complete'] and facts['validation']['passed']
 OUT.mkdir(parents=True,exist_ok=True);R=facts['results']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 active={'ESCA':['pair_ESCA_STAD','pair_ESCA_COAD','triple','global_selected','target_selected'],'STAD':['pair_ESCA_STAD','pair_STAD_COAD','triple','global_selected','target_selected'],'COAD':['pair_ESCA_COAD','pair_STAD_COAD','triple','global_selected','target_selected']}
 labels={'pair_ESCA_STAD':'ESCA + STAD','pair_ESCA_COAD':'ESCA + COAD','pair_STAD_COAD':'STAD + COAD','triple':'Three tasks','global_selected':'Global selection','target_selected':'Target-specific selection'}
 colors=['#4d7188','#b16e5f'];markers=['o','s']
 fig,axes=plt.subplots(3,1,figsize=(6.5,7.05),sharex=True)
 for ci,(c,models) in enumerate(active.items()):
  ax=axes[ci]
  for si,seed in enumerate(['42','2026']):
   for j,m in enumerate(models):
    v=R[seed][c][m]['c_index_vs_single'];x=v['difference'];lo,hi=v['ci95'];y=4-j+(0.13 if si==0 else -.13)
    ax.hlines(y,lo,hi,color=colors[si],lw=1)
    ax.vlines([lo,hi],y-.045,y+.045,color=colors[si],lw=1)
    ax.plot(x,y,marker=markers[si],color=colors[si],markersize=4,ls='',label=('Seed '+seed) if ci==0 and j==0 else None)
  ax.axvline(0,color='#7d7d7d',lw=.8,ls='--');ax.set_yticks(range(5));ax.set_yticklabels([labels[m] for m in models][::-1],fontsize=9)
  ax.set_ylim(-.55,4.65);ax.grid(axis='x',alpha=.17);ax.set_title(chr(97+ci)+'  '+c,loc='left',fontweight='bold',fontsize=11)
 axes[-1].set_xlabel('C-index difference from matched single task')
 axes[0].legend(loc='lower right',bbox_to_anchor=(1.,1.015),fontsize=8,frameon=False,ncol=2)
 fig.subplots_adjust(left=.31,right=.97,top=.965,bottom=.085,hspace=.36)
 for ext in ['png','pdf']:fig.savefig(OUT/('Figure2_sharing_differences.'+ext),dpi=400)
 plt.close(fig)
 fig,axes=plt.subplots(3,3,figsize=(6.8,7.14),sharex=True,sharey=True)
 for ci,(c,models) in enumerate(active.items()):
  for j,m in enumerate(models[:3]):
   ax=axes[ci,j]
   for si,seed in enumerate([42,2026]):
    v=sorted([x for x in facts['strength_curves'] if x['seed']==seed and x['cancer']==c and x['graph']==m],key=lambda x:x['mu'])
    y=[x['c_index']-R[str(seed)][c]['single']['c_index'] for x in v]
    ax.plot([x['mu'] for x in v],y,marker=markers[si],color=colors[si],lw=1,markersize=3,label='Seed '+str(seed))
   ax.axhline(0,color='#7d7d7d',ls='--',lw=.7);ax.set_xscale('log');ax.grid(alpha=.15);ax.tick_params(labelsize=8)
   ax.set_title(c+'\n'+labels[m],fontsize=9)
   if j==0:ax.set_ylabel('ΔC-index',fontsize=9)
   if ci==2:ax.set_xlabel('Incident coupling μ',fontsize=9)
 handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=2,fontsize=9,frameon=False)
 fig.subplots_adjust(left=.12,right=.98,top=.93,bottom=.11,hspace=.38,wspace=.20)
 for ext in ['png','pdf']:fig.savefig(OUT/('FigureS1_strength_response.'+ext),dpi=400)
 plt.close(fig)
 print('Rendered new forest plot and all-strength response curves')
if __name__=='__main__':main()
