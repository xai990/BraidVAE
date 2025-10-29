import pandas as pd 
import torch as th 
import scanpy as sc
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from bin.utils import extract_embeddings,visualize_embeddings
from bin.dataset import load_txt_to_anndata, align_genes,preprocess_anndata
from bin import logger, dist_util
import argparse
from omegaconf import OmegaConf
from bin.train_utils import train_multistage_vaes
import glob
import os 



def main(args):
    """Main function to run the complete pipeline"""
    
    print("Loading and preprocessing data...")
    
    # read the data from the txt file 
    # Define file paths
    
    # Configure the logger with specified directory
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    # Setup distributed training environment
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    # Create and merge configuration files
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    
    # load data 
    logger.info(f"The args gene set is: {args.gene_set}")

    tr = load_txt_to_anndata(config.data.train_path, config.data.train_label_path,
                             label_col=config.data.label_col)
    te = load_txt_to_anndata(config.data.test_path, config.data.test_label_path,
                             label_col=config.data.label_col)
    
    # 2) Align genes
    
    # logger.log(f"The info of tr:{tr.obs}")
    # logger.log(f"The info of tr:{te.obs}")
    # 3) Preprocess.
    tr_filter = preprocess_anndata(tr, nan_policy=config.data.filter)
    te_filter = preprocess_anndata(te, nan_policy=config.data.filter)
    tr_filter, te_filter = align_genes(tr_filter, te_filter)
    # 4) Train one VAE per stage
    summary = train_multistage_vaes(
        tr_filter, te_filter,
        label_key=config.data.label_col,      # set to None if you want a plain (non-conditional) VAE
        latent_dim=config.train.latent_dim,
        hidden_dims=[512, 256, 128],
        batch_size=config.train.batch_size,
        epochs=config.train.epochs,
    )
    logger.log("\nAggregating all latent embeddings across stages...")
    all_latent_files = glob.glob("results_multistage/stage_*/latent_all.tsv")
    dfs = []
    for f in all_latent_files:
        df = pd.read_csv(f, sep="\t", index_col=0)
        df["stage"] = os.path.basename(os.path.dirname(f)).replace("stage_", "")
        dfs.append(df)
    logger.log(f"The dfs is:{dfs}")
    if dfs:
        df_all = pd.concat(dfs, axis=0)
        logger.log(f"Combined {len(df_all)} samples from {len(dfs)} stages.")

        # visualize global UMAP
        visualize_embeddings(
            df_all.iloc[:, :100].values,  # exclude meta columns
            labels=df_all[config.data.label_col].values if config.data.label_col in df_all.columns else None,
            label_names=np.unique(df_all[config.data.label_col].values) if config.data.label_col in df_all.columns else None,
            save_path="results_multistage/global_umap.png"
        )


def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            "samples":124,
            "drop_fraction":0,
            "random": False,
            "label_col": "label",
            "filter":"drop"
        },
        "train":{
            "batch_size":16,
            "latent_dim":200,
            "epochs":200,
           
        }
        
    }
   
    return defaults  

if __name__ == "__main__":
    # Run the main pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    args = parser.parse_args()
    main(args)