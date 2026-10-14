"""Restore the exact 4,098-feature model-input snapshot; no inferred raw-data transform."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    resources=ROOT/'resources'; dest=ROOT/'datasets_breast'
    if not (resources/'input_checksums.json').exists():
        raise SystemExit('Historical input files are not distributed. See docs/DATA.md for source preparation.')
    records=json.loads((resources/'input_checksums.json').read_text())
    for record in records:
        path=resources/record['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
    manifest=pd.read_csv(resources/'sample_manifest.tsv',sep='\t',index_col='sample_id')
    if not manifest.index.is_unique or len(manifest)!=1138:raise ValueError('Invalid sample manifest')
    genes=(resources/'model_genes.txt').read_text().splitlines()
    dest.mkdir(exist_ok=True)
    for split in ['train','test']:
        frame=pd.read_csv(resources/f'model_input_{split}.tsv.gz',sep='\t',index_col=0)
        ids=manifest.index[manifest.split==split]
        if list(frame.index)!=list(ids) or list(frame.columns)!=genes:raise ValueError('Sample/gene order mismatch')
        frame.to_csv(dest/f'breast_{split}_GEM_transpose.txt',sep='\t')
        manifest.loc[ids,['label']].to_csv(dest/f'{split}_stage_labels_new_updated.txt',sep='\t',header=False)
        print(split,frame.shape,manifest.loc[ids,'label'].value_counts().to_dict())
    (dest/'hallmark_allgene_transposed.txt').write_text((resources/'hallmark_allgene_transposed.txt').read_text())
    (ROOT/'gene_names.txt').write_text('\n'.join(genes)+'\n')
    print('Prepared exact selected-feature inputs. Historical test split is used for VAE early stopping.')

if __name__=='__main__':main()
