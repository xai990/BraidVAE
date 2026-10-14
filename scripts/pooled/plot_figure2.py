"""Redraw expression, saved pooled VAE, and saved aligned VAE; no training."""
from pathlib import Path
import os, sys, types, shutil, json
OUT=Path(__file__).resolve().parent
BASE=Path(__file__).resolve().parents[2]
ROOT=Path(__file__).resolve().parents[2]
os.environ['NUMBA_CACHE_DIR']=str(OUT/'cache')
os.environ['MPLCONFIGDIR']=str(OUT/'cache/matplotlib')
sys.dont_write_bytecode=True
pkg=types.ModuleType('tsbrca_bin'); pkg.__path__=[str(ROOT/'bin')]; sys.modules['tsbrca_bin']=pkg
(Path(__file__).resolve().parents[2]/'Figures').mkdir(exist_ok=True)
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tsbrca_bin.dataset import load_txt_to_anndata,preprocess_anndata,align_genes
ref=pd.read_csv(ROOT/'resources/latent_all_stages_aligned.tsv',sep='\t',index_col=0)
pooled=pd.read_csv(OUT/'pooled_latent.tsv',sep='\t',index_col=0).loc[ref.index]
parts=[]
for split in ['train','test']:
 a=load_txt_to_anndata(str(ROOT/f'datasets_breast/breast_{split}_GEM_transpose.txt'),str(ROOT/f'datasets_breast/{split}_stage_labels_new_updated.txt'),gene_list_path=str(ROOT/'datasets_breast/hallmark_allgene_transposed.txt'))
 ids=pooled.index[pooled['split']==split]
 a=preprocess_anndata(a[ids].copy(),nan_policy='replace'); parts.append(a)
tr,te=align_genes(*parts)
expr=pd.concat([pd.DataFrame(a.X,index=a.obs_names,columns=a.var_names) for a in [tr,te]]).loc[ref.index]
labels=ref['label'].to_numpy()
assert np.array_equal(labels,pooled['label'].to_numpy())
colors={'normal_GTEX':'#1f77b4','normal_TCGA':'#ff7f0e','Stage_I':'#2ca02c','Stage_II':'#d62728','Stage_III':'#9467bd'}
zs=[f'z{i}' for i in range(1,33)]
coords=[]
for name,x in [('original',expr.to_numpy()),('single_vae',pooled[zs].to_numpy()),('aligned',ref[zs].to_numpy())]:
 a=sc.AnnData(x)
 if name=='original': sc.tl.pca(a,n_comps=50,random_state=0)
 sc.pp.neighbors(a,n_neighbors=15,use_rep='X_pca' if name=='original' else 'X',random_state=0)
 sc.tl.umap(a,min_dist=0.5,random_state=42)
 xy=a.obsm['X_umap']; coords.append(xy)
 pd.DataFrame({'sample_id':ref.index,'UMAP1':xy[:,0],'UMAP2':xy[:,1],'label':labels}).to_csv(OUT/f'figure2_{name}_coordinates.tsv',sep='\t',index=False)

def draw(ax,xy,title):
 for label,color in colors.items():
  m=labels==label
  ax.scatter(xy[m,0],xy[m,1],c=color,s=15,alpha=0.8,label=label)
 ax.set_title(title,fontsize=12,pad=10)
 ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
 ax.tick_params(labelsize=9, labelbottom=False, labelleft=False)

fig,axes=plt.subplots(1,3,figsize=(17,5.4))
for ax,xy,title in zip(axes,coords,['(A) Original expression (PCA + UMAP)','(B) Single pooled VAE','(C) Aligned stage-specific VAEs']):draw(ax,xy,title)
handles=[Line2D([],[],marker='o',linestyle='',color=c,alpha=0.8,markersize=5,label=l) for l,c in colors.items()]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5,0.01))
fig.tight_layout(rect=[0,0.09,1,1])
backup=BASE/'Figures/Fig2_original_dataplot_before_comparison.png'
if not backup.exists() and (BASE/'Figures/Fig2_original_dataplot.png').exists():shutil.copy2(BASE/'Figures/Fig2_original_dataplot.png',backup)
fig.savefig(BASE/'Figures/Fig2_original_dataplot.png',dpi=300,bbox_inches='tight')
fig.savefig(BASE/'Figures/Fig2_three_way_comparison.pdf',bbox_inches='tight')
plt.close(fig)
for xy,name,title in zip(coords,['original_restyled','single_vae','aligned_restyled'],['Original expression (PCA + UMAP)','Single pooled VAE','Aligned stage-specific VAEs']):
 fig,ax=plt.subplots(figsize=(8,7));draw(ax,xy,title)
 ax.legend(loc='upper right');fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=200);plt.close(fig)
(OUT/'figure2_settings.json').write_text(json.dumps(dict(n_samples=len(ref),expression_genes=expr.shape[1],pca_components=50,neighbors=15,min_dist=0.5,umap_seed=42,neighbors_seed=0,point_size=15,alpha=0.8,colors=colors,note='No model training. Original expression regenerated for same sample cohort. Figure 3 source settings used for all UMAPs.'),indent=2))
print('Figure 2 complete',flush=True)
