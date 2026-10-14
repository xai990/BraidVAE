"""Exploratory pooled unconditional VAE using the existing tsbrca protocol."""
import os
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ['NUMBA_CACHE_DIR'] = str(OUT / 'cache')
os.environ['MPLCONFIGDIR'] = str(OUT / 'cache' / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, types, json, random, hashlib, importlib.metadata
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
pkg = types.ModuleType('tsbrca_bin')
pkg.__path__ = [str(ROOT / 'bin')]
sys.modules['tsbrca_bin'] = pkg
(Path(__file__).resolve().parents[2]/'Figures').mkdir(exist_ok=True)
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.metrics import silhouette_score
from tsbrca_bin.dataset import load_txt_to_anndata, preprocess_anndata, align_genes, PerGeneMinMaxScaler, make_loader
from tsbrca_bin.model import ConditionalVAE
from tsbrca_bin.train_utils import train_vae_with_test_set
from tsbrca_bin.utils import extract_embeddings, STAGE_COLORS

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)
ref = pd.read_csv(ROOT / 'resources/latent_all_stages.tsv', sep='\t', index_col=0)
aligned = pd.read_csv(ROOT / 'resources/latent_all_stages_aligned.tsv', sep='\t', index_col=0)
assert ref.index.is_unique and set(ref.index) == set(aligned.index)
data = []
for split in ['train', 'test']:
    adata = load_txt_to_anndata(str(ROOT / f'datasets_breast/breast_{split}_GEM_transpose.txt'), str(ROOT / f'datasets_breast/{split}_stage_labels_new_updated.txt'), gene_list_path=str(ROOT / 'datasets_breast/hallmark_allgene_transposed.txt'))
    ids = ref.index[ref['split'] == split]
    assert set(ids).issubset(adata.obs_names), 'Historical sample IDs missing from input'
    adata = adata[ids].copy()
    assert np.array_equal(adata.obs['label'].to_numpy(), ref.loc[ids, 'label'].to_numpy()), 'Labels changed'
    adata = preprocess_anndata(adata, nan_policy='replace')
    assert np.isfinite(adata.X).all()
    data.append(adata)
tr, va = align_genes(*data)
assert not set(tr.obs_names) & set(va.obs_names)
assert tr.n_vars == 4098
scaler = PerGeneMinMaxScaler()
scaler.fit(tr.X)
np.savez(OUT / 'scaler.npz', min=scaler.min_, max=scaler.max_, range=scaler.range_)
pd.Series(tr.var_names).to_csv(OUT / 'genes.tsv', index=False, header=False)
# Labels are carried for later plotting only. n_classes=0 ignores them entirely.
tr_loader, _ = make_loader(tr, 16, True, label_key='label', scaler=scaler)
va_loader, _ = make_loader(va, 16, False, label_key='label', scaler=scaler)
model = ConditionalVAE(4098, latent_dim=32, hidden_dims=[1024,256,64], n_classes=0)
assert not model.is_conditional
manifest = dict(seed=42, train_samples=tr.n_obs, validation_samples=va.n_obs, genes=tr.n_vars, latent_dim=32, hidden_dims=[1024,256,64], batch_size=16, learning_rate=0.001, max_epochs=200, patience=10, kl_warmup=50, device='cpu', conditioning=False, encoder_dropout=0.5, preprocessing='Existing preprocess_anndata(replace), pooled training-only per-gene minmax to [-1,1]', validation_note='Original test split is used for early stopping, not an independent test.', comparison_note='Historical multi-VAE uses stage labels for model assignment and stage-specific scaling; exploratory pipeline comparison, not a controlled architecture-only ablation.', versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scanpy','umap-learn','scikit-learn']})
manifest['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'bin/model.py',ROOT/'bin/train_utils.py',ROOT/'bin/dataset.py',ROOT/'resources/latent_all_stages.tsv',ROOT/'resources/latent_all_stages_aligned.tsv']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Training:', manifest, flush=True)
history = train_vae_with_test_set(model,tr_loader,va_loader,n_epochs=200,device='cpu',ckpt_path=str(OUT/'best_vae.pth'))
(OUT/'history.json').write_text(json.dumps(history,indent=2))
frames=[]
for split, adata in [('train',tr),('test',va)]:
    loader,_=make_loader(adata,16,False,label_key='label',scaler=scaler)
    z,names,_=extract_embeddings(model,loader,device='cpu')
    frame=pd.DataFrame(z,index=names,columns=[f'z{i}' for i in range(1,33)])
    frame['label']=adata.obs.loc[names,'label'].values
    frame['split']=split
    frames.append(frame)
pooled=pd.concat(frames).loc[ref.index]
pooled.to_csv(OUT/'pooled_latent.tsv',sep='\t')
spaces={'Pooled VAE':pooled,'Stage-specific VAEs (unaligned)':ref,'Stage-specific VAEs + MNN/Procrustes':aligned.loc[ref.index]}
fig,axes=plt.subplots(1,3,figsize=(17,5))
metrics=[]
for ax,(name,df) in zip(axes,spaces.items()):
    z=df[[f'z{i}' for i in range(1,33)]].to_numpy()
    assert np.isfinite(z).all()
    labels=df['label'].to_numpy()
    xy=UMAP(n_neighbors=30,min_dist=0.2,random_state=42,n_jobs=1).fit_transform(z)
    slug=['pooled','multi_unaligned','multi_aligned'][len(metrics)]
    pd.DataFrame({'sample_id':df.index,'UMAP1':xy[:,0],'UMAP2':xy[:,1],'label':labels}).to_csv(OUT/f'{slug}_umap.tsv',sep='\t',index=False)
    for label,color in STAGE_COLORS.items():
        mask=labels==label
        if mask.any(): ax.scatter(xy[mask,0],xy[mask,1],s=9,alpha=0.7,c=color,label=label)
    ax.set_title(name,fontsize=11)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    tumor=np.isin(labels,['Stage_I','Stage_II','Stage_III'])
    metrics.append({'model':name,'n_samples':len(df),'silhouette_32d_all_groups':float(silhouette_score(z,labels)),'silhouette_32d_tumor_stages':float(silhouette_score(z[tumor],labels[tumor]))})
axes[-1].legend(fontsize=8,loc='best')
fig.suptitle('Exploratory comparison: identical samples and UMAP settings; seed 42',fontsize=13)
fig.tight_layout()
fig.savefig(OUT/'umap_comparison.png',dpi=180)
plt.close(fig)
pd.DataFrame(metrics).to_csv(OUT/'descriptive_metrics.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(history['train_recon'],label='Training reconstruction MSE')
ax.plot(history['test_recon'],label='Validation reconstruction MSE')
ax.set_xlabel('Epoch (zero-based)'); ax.legend(); fig.tight_layout()
fig.savefig(OUT/'training_curve.png',dpi=160)
print('DONE', json.dumps(metrics),flush=True)
