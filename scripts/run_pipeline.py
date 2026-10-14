"""Explicit pipeline runner. Defaults to dry run; --execute performs analysis."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--gene-selection',choices=['std','elbow'],default='elbow')
    args=p.parse_args()
    steps=[['brcavae.py','--config','configs/config.yaml'],
           ['align_latent_spaces_mnn.py'],['plot_umap_aligned.py'],['calc_supervised_pseudotime_aligned.py'],
           ['train_cascaded.py','--data_path','latent_all_stages_aligned.tsv','--outdir','results_cascaded_aligned'],
           ['analyze_cascaded_genes_aligned.py','--gene-selection',args.gene_selection],
           ['plot_informer_dim_histogram_aligned.py'],['plot_gene_dimension_heatmap_aligned.py']]
    for step in steps:
        print(' '.join([sys.executable,*step]),flush=True)
        if args.execute:subprocess.run([sys.executable,*step],cwd=ROOT,check=True)

if __name__=='__main__':main()
