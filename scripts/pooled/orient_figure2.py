"""Rigidly rotate saved UMAP coordinates for display; no fitting or training."""
from pathlib import Path
import os,json
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'cache/matplotlib')
(Path(__file__).resolve().parents[2]/'Figures').mkdir(exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
colors={'normal_GTEX':'#1f77b4','normal_TCGA':'#ff7f0e','Stage_I':'#2ca02c','Stage_II':'#d62728','Stage_III':'#9467bd'}
fig,axes=plt.subplots(1,3,figsize=(17,5.4))
rotations={}
for ax,name,title in zip(axes,['original','single_vae','aligned'],['(A) Original expression (PCA + UMAP)','(B) Single pooled VAE','(C) Aligned stage-specific VAEs']):
 df=pd.read_csv(OUT/f'figure2_{name}_coordinates.tsv',sep='\t')
 xy=df[['UMAP1','UMAP2']].to_numpy()
 normal=df.label.isin(['normal_GTEX','normal_TCGA']).to_numpy()
 delta=xy[normal].mean(0)-xy[~normal].mean(0)
 angle=np.pi/4-np.arctan2(delta[1],delta[0])
 c,s=np.cos(angle),np.sin(angle)
 rotation=np.array([[c,-s],[s,c]])
 assert np.allclose(rotation.T@rotation,np.eye(2))
 centered=xy-xy.mean(0)
 rotated=centered@rotation.T
 assert np.allclose(np.linalg.norm(np.diff(xy,axis=0),axis=1),np.linalg.norm(np.diff(rotated,axis=0),axis=1))
 df[['UMAP1','UMAP2']]=rotated
 df.to_csv(OUT/f'figure2_{name}_oriented_coordinates.tsv',sep='\t',index=False)
 rotations[name]={'rotation_degrees':float(np.degrees(angle)),'matrix':rotation.tolist()}
 for label,color in colors.items():
  mask=(df.label==label).to_numpy()
  ax.scatter(rotated[mask,0],rotated[mask,1],c=color,s=15,alpha=0.8,label=label)
 ax.set_title(title,fontsize=12,pad=10)
 ax.set_xlabel('UMAP 1 (rotated)');ax.set_ylabel('UMAP 2 (rotated)')
 center=(rotated.min(0)+rotated.max(0))/2
 half=float(np.ptp(rotated,axis=0).max())*0.55
 ax.set_xlim(center[0]-half,center[0]+half); ax.set_ylim(center[1]-half,center[1]+half)
 ax.set_aspect('equal',adjustable='box')
 ax.tick_params(labelsize=9, labelbottom=False, labelleft=False)
handles=[Line2D([],[],marker='o',linestyle='',color=c,alpha=0.8,markersize=5,label=l) for l,c in colors.items()]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5,0.01))
fig.tight_layout(rect=[0,0.09,1,1])
fig.savefig(Path(__file__).resolve().parents[2]/'Figures/Fig2_original_dataplot.png',dpi=300,bbox_inches='tight')
fig.savefig(Path(__file__).resolve().parents[2]/'Figures/Fig2_three_way_comparison.pdf',bbox_inches='tight')
(OUT/'figure2_display_rotations.json').write_text(json.dumps(rotations,indent=2))
print('Saved rigidly rotated Figure 2; distance preservation checked.')
