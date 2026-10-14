"""Prepare Figshare matrices with the author-reported log2(x+1) transform.

Requires a user-supplied sample manifest and ordered gene list.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,default=ROOT/'raw/figshare_5330593_v2')
    p.add_argument('--outdir',type=Path,default=ROOT/'processed/figshare_log2p1')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--genes',type=Path,required=True)
    args=p.parse_args()
    manifest=pd.read_csv(args.manifest,sep='\t',index_col=0)
    genes=args.genes.read_text().splitlines()
    if not manifest.index.is_unique or manifest.empty:
        raise ValueError('Manifest must have unique sample IDs and at least one row')
    if not {'label','split'}.issubset(manifest.columns):
        raise ValueError('Manifest requires label and split columns')
    if set(manifest['split']) != {'train','test'}:
        raise ValueError('Manifest must contain train and test splits only')
    labels={'normal_GTEX','normal_TCGA','Stage_I','Stage_II','Stage_III'}
    if not set(manifest['label']).issubset(labels):
        raise ValueError('Unsupported or missing stage labels')
    if not genes or len(set(genes)) != len(genes) or any(not g.strip() for g in genes):
        raise ValueError('Gene list must contain unique nonempty symbols')
    groups=[]
    for name in ['brca-rsem-fpkm-tcga-t.txt.gz','brca-rsem-fpkm-tcga.txt.gz','breast-rsem-fpkm-gtex.txt.gz']:
        df=pd.read_csv(args.raw_dir/name,sep='\t')
        if df.Hugo_Symbol.duplicated().any():raise ValueError(f'Duplicate gene symbols in {name}; resolve explicitly')
        groups.append(df.set_index('Hugo_Symbol').drop(columns='Entrez_Gene_Id').T)
    raw=pd.concat(groups)
    if not raw.index.is_unique:raise ValueError('Duplicate sample IDs')
    missing=manifest.index.difference(raw.index)
    if len(missing):raise ValueError(f'Missing sample IDs: {list(missing[:10])}')
    selected=raw.loc[manifest.index,genes]
    if (selected.to_numpy()<0).any():raise ValueError('Negative source expression')
    reconstructed=np.log2(selected+1)
    args.outdir.mkdir(parents=True,exist_ok=True)
    report={'transform':'log2(x + 1)','source':'10.6084/m9.figshare.5330593.v2','splits':{}}
    for split in ['train','test']:
        ids=manifest.index[manifest.split==split]
        x=reconstructed.loc[ids]
        x.to_csv(args.outdir/f'breast_{split}_GEM_transpose.txt',sep='\t')
        manifest.loc[ids,['label']].to_csv(args.outdir/f'{split}_stage_labels_new_updated.txt',sep='\t',header=False)
        report['splits'][split]={'samples':len(ids),'genes':len(genes)}
    (args.outdir/'hallmark_allgene_transposed.txt').write_text('hallmark\t'+'\t'.join(genes)+'\n')
    (args.outdir/'preprocessing.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(f'Prepared source matrices and gene list in {args.outdir}')

if __name__=='__main__':main()
