import pandas as pd 
import torch as th 
import scanpy as sc
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from bin.utils import extract_embeddings,visualize_embeddings,origin_plot,get_active_latent_dims
from bin.dataset import load_txt_to_anndata, align_genes,preprocess_anndata, balance_anndata_by_label, balance_latent_df
from bin import logger, dist_util
import argparse
from omegaconf import OmegaConf
from bin.train_utils import train_multistage_vaes, train_single_vae
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
    logger.info(config)
    # load data 
    logger.info(f"The args gene set is: {config.data.gene_list}")

    tr = load_txt_to_anndata(config.data.train_path, config.data.train_label_path,
                             label_col=config.data.label_col, gene_list_path = config.data.gene_list)
    te = load_txt_to_anndata(config.data.test_path, config.data.test_label_path,
                             label_col=config.data.label_col, gene_list_path = config.data.gene_list)
    
    # 2) Align genes
    
    logger.log(f"The info of tr:{tr.X.shape}")
    logger.log(f"The info of tr:{te.X.shape}")
    
    # 3) Preprocess.
    tr_filter = preprocess_anndata(tr, nan_policy=config.data.filter)
    te_filter = preprocess_anndata(te, nan_policy=config.data.filter)
    tr_filter, te_filter = align_genes(tr_filter, te_filter)
    # optinal balance the sample size
    # tr_filter = balance_anndata_by_label(tr_filter,label_key="label",random_state=42,)
    # te_filter = balance_anndata_by_label(te_filter,label_key="label",random_state=42,)
    # optinal visulize the original data
    origin_plot(tr_filter)
    
    # 4) Train one VAE per stage
    if config.train.multivae:
        logger.log("Train multi vaes for each stage data")
        summary = train_multistage_vaes(
            tr_filter, te_filter,
            label_key=config.data.label_col,      # set to None if you want a plain (non-conditional) VAE
            latent_dim=config.train.latent_dim,
            hidden_dims=config.train.hidden_dims,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            outdir=config.data.dir_out
        )
        logger.log("\nAggregating all latent embeddings across stages...")
        all_latent_files = glob.glob(os.path.join(config.data.dir_out,"stage_*/latent_all.tsv"))
        dfs = []
        for f in all_latent_files:
            df = pd.read_csv(f, sep="\t", index_col=0)
            df["stage"] = os.path.basename(os.path.dirname(f)).replace("stage_", "")
            dfs.append(df)
        logger.log(f"The dfs is:{dfs}")
        if dfs:
            df_all = pd.concat(dfs, axis=0)
            logger.log(f"Combined {len(df_all)} samples from {len(dfs)} stages.")
            latent_path = os.path.join(config.data.dir_out, "latent_all.tsv") 
            df_all.to_csv(latent_path, sep="\t")                    
            logger.log(f"[save] Latent embeddings saved to {latent_path}") 
            if config.data.balance is not None:
                df_all = balance_latent_df(
                    df_all,
                    label_col=config.data.label_col,
                    random_state=42,
                )
            else:
                logger.log(
                    f"[warn] label_col '{config.data.label_col}' not found in latent DF. "
                    "Skipping latent balancing."
                )
            # visualize global UMAP
            visualize_embeddings(
                df_all.iloc[:, :-3].values,  # exclude meta columns
                labels=df_all[config.data.label_col].values if config.data.label_col in df_all.columns else None,
                label_names=np.unique(df_all[config.data.label_col].values) if config.data.label_col in df_all.columns else None,
                save_path= os.path.join(config.data.dir_out, "global_umap.png")
            )
            logger.log(f"The results are save to:{config.data.dir_out}")
    else:
        logger.log("Train single vae using all stage data")
        # Train a single VAE on all samples (no per-stage split)
        single_summary = train_single_vae(
            train_adata=tr_filter,
            test_adata=te_filter,
            label_key=config.data.label_col,
            latent_dim=config.train.latent_dim,
            hidden_dims=config.train.hidden_dims,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            early_stop=10,
            warmup_epochs=50,
            outdir=config.data.dir_out,
            run_name="single_vae",
        )
    # activate_latent_dim = get_active_latent_dims(df_all,outdir=config.data.dir_out)
    # df_activate = df_all[activate_latent_dim]
    # visualize_embeddings(
    #     df_activate.values,  # exclude meta columns
    #     labels=df_all[config.data.label_col].values if config.data.label_col in df_all.columns else None,
    #     label_names=np.unique(df_all[config.data.label_col].values) if config.data.label_col in df_all.columns else None,
    #     save_path= os.path.join(config.data.dir_out, "globaldrop_umap.png")
    # )
    
    logger.log("process complete")


def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results_multistagecoad",
            "gene_selection": None,
            "samples":124,
            "drop_fraction":0,
            "random": False,
            "label_col": "label",
            "filter":"drop",
            "gene_list": None,
            "balance": None
        },
        "train":{
            "batch_size":16,
            "latent_dim":8,
            "epochs":200,
            "multivae":True,
            "hidden_dims":[256, 128, 64]
        },
        
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