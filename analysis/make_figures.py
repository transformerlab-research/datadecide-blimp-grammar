#!/usr/bin/env python3
"""Regenerate the three paper figures from ../data and the DataDecide eval bundle.
Deps: pandas numpy scipy pyarrow matplotlib requests.  Outputs PNGs next to this file."""
import os, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qa_join import qa_ranking, DATA, spearman_ci

def isfilt(r): return ('qc-fw' in r) or ('fw2' in r) or ('fw3' in r) or (r=='fineweb-edu') or ('qc-orig' in r)
def load(size, csv): 
    bl=pd.read_csv(os.path.join(DATA,csv)).set_index('recipe')
    return pd.DataFrame({'blimp':bl.macro5_no_detn,'qa':qa_ranking(size)}).dropna(), bl
j9,bl9=load('90M','blimp-90m-v2-full.csv'); j2,_=load('20M','blimp-20m-full.csv')
r9,p9,lo9,hi9=spearman_ci(j9.blimp,j9.qa); r2,p2,lo2,hi2=spearman_ci(j2.blimp,j2.qa)

j=j9.copy(); j['f']=[isfilt(r) for r in j.index]; plt.figure(figsize=(5.2,3.6))
for f,c,l in [(False,'#3b6ea5','other recipes'),(True,'#c0392b','classifier-filtered')]:
    s=j[j.f==f]; plt.scatter(s.qa*100,s.blimp*100,c=c,s=34,label=l,edgecolor='white',lw=.5,zorder=3)
z=np.polyfit(j.qa*100,j.blimp*100,1); xs=np.array([j.qa.min()*100,j.qa.max()*100]); plt.plot(xs,np.polyval(z,xs),'--',c='#555',lw=1)
plt.xlabel('DataDecide QA (olmes_10 macro, %)'); plt.ylabel('BLiMP grammar (5-par macro, %)')
plt.title(f'90M:  Spearman rho = {r9:+.2f}  (p={p9:.3f})'); plt.legend(fontsize=8); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig('fig1_scatter_90m.png',dpi=170); plt.close()

plt.figure(figsize=(3.6,3.2)); xs=[0,1]; rs=[r2,r9]
plt.errorbar(xs,rs,yerr=[[rs[0]-lo2,rs[1]-lo9],[hi2-rs[0],hi9-rs[1]]],fmt='o',c='#c0392b',capsize=5,ms=8,lw=1.5)
plt.axhline(0,c='#999',lw=.8,ls=':'); plt.xticks(xs,['20M','90M']); plt.ylim(-1,.4); plt.xlim(-.5,1.5)
plt.ylabel('Spearman rho (grammar vs QA)'); plt.title('Anti-correlation emerges with scale'); plt.grid(alpha=.25,axis='y'); plt.tight_layout(); plt.savefig('fig2_rho_by_size.png',dpi=170); plt.close()

base=bl9.loc['dclm-baseline','macro6']
g={'DCLM +\nFineWeb filter':['dclm-baseline-qc-fw-3p','dclm-baseline-qc-7p-fw3','dclm-baseline-qc-fw-10p','dclm-baseline-qc-7p-fw2'],'DCLM +\nplain QC':['dclm-baseline-qc-10p','dclm-baseline-qc-20p'],'Falcon+CC +\ndefault/tulu QC':['falcon-and-cc-qc-10p','falcon-and-cc-qc-tulu-10p'],'FineWeb-Edu\n(corpus)':['fineweb-edu']}
labs=list(g); vals=[np.mean([(bl9.loc[r,'macro6']-base)*100 for r in g[k] if r in bl9.index]) for k in labs]
plt.figure(figsize=(5.2,3.2)); plt.bar(range(len(labs)),vals,color=['#c0392b','#3b6ea5','#3b6ea5','#c0392b'],edgecolor='white')
plt.axhline(0,c='#333',lw=.8); plt.xticks(range(len(labs)),labs,fontsize=7.5); plt.ylabel('delta BLiMP macro vs\nDCLM-baseline (pp)')
plt.title('The cost tracks the classifier, not filtering'); plt.grid(alpha=.25,axis='y'); plt.tight_layout(); plt.savefig('fig3_classifier_bars.png',dpi=170); plt.close()
print(f"figures written. 90M rho={r9:+.3f}, 20M rho={r2:+.3f}")
