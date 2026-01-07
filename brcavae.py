import pandas as pd 
import torch as th 
import scanpy as sc
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from bin.utils import extract_embeddings, visualize_embeddings, origin_plot, get_active_latent_dims
from bin.dataset import load_txt_to_anndata, align_genes, preprocess_anndata, balance_anndata_by_label, balance_latent_df
from bin import logger, dist_util
import argparse
from omegaconf import OmegaConf
# Importing all necessary training functions
from bin.train_utils import (
    train_multistage_vaes, 
    train_single_vae, 
    train_trajectory_vae, 
    train_gene_transformer, 
    train_latent_transformer
)
import glob
import os
from bin.utils import seed_everything


def main(args):
    """Main function to run the complete pipeline"""
    
    print("Loading and preprocessing data...")
    
    # Configure the logger with specified directory
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    
    # Setup distributed training environment
    dist_util.setup_dist()
    seed_everything(42)
    logger.log(f"device information:{dist_util.dev()}")
    
    # Create and merge configuration files
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    
    # Override config with args if provided
    if args.model_type:
        config.train.model_type = args.model_type
        
    logger.info(config)
    
    # =========================================================
    # 1) Load Data
    # =========================================================
    logger.info(f"The args gene set is: {config.data.gene_list}")

    tr = load_txt_to_anndata(config.data.train_path, config.data.train_label_path,
                             label_col=config.data.label_col, gene_list_path = config.data.gene_list)
    te = load_txt_to_anndata(config.data.test_path, config.data.test_label_path,
                             label_col=config.data.label_col, gene_list_path = config.data.gene_list)
    
    logger.log(f"The info of tr:{tr.X.shape}")
    logger.log(f"The info of te:{te.X.shape}")
    
    # =========================================================
    # 2) Preprocess & Align
    # =========================================================
    tr_filter = preprocess_anndata(tr, nan_policy=config.data.filter)
    te_filter = preprocess_anndata(te, nan_policy=config.data.filter)
    tr_filter, te_filter = align_genes(tr_filter, te_filter)
    
    # Optional: visualize original data
    origin_plot(tr_filter)
    
    # =========================================================
    # 3) Train Logic Switch
    # =========================================================
    
    # A. DIRECT TRANSFORMER (Genes -> Class)
    if config.train.model_type == "transformer":
        logger.log("Train Gene Transformer (Informer-like Encoder) on RAW GENES")
        summary = train_gene_transformer(
            train_adata=tr_filter,
            test_adata=te_filter,
            label_key=config.data.label_col,
            top_k=config.train.top_k,
            d_model=config.train.d_model,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            lr=config.train.lr if 'lr' in config.train else 1e-3,
            outdir=config.data.dir_out
        )
        logger.log("Transformer training complete.")

    # B. TRANSFORMER ON LATENT (Genes -> VAE -> Latent -> Transformer -> Class)
    elif config.train.model_type == "transformer_on_latent":
        logger.log("Pipeline: Multi-VAE -> Latent Aggregation -> Transformer Distillation")
        
        # Step 1: Train Multi-VAEs (One per stage)
        df_all, summary = train_multistage_vaes(
            tr_filter, te_filter,
            label_key=config.data.label_col,
            latent_dim=config.train.latent_dim,
            hidden_dims=config.train.hidden_dims,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            outdir=config.data.dir_out
        )
        
        # Step 2: Train Transformer on the aggregated Latents
        logger.log("Training Transformer on VAE Latents...")
        df_distilled = train_latent_transformer(
            latent_df=df_all,
            label_key=config.data.label_col,
            d_model=64,             # Internal dim for the small transformer
            nhead=4,
            num_layers=2,
            keep_ratio=0.5,         # Keep top 50% of latent dims
            batch_size=32,
            epochs=100,
            outdir=os.path.join(config.data.dir_out, "transformer_latent_results")
        )
        logger.log("Latent Transformer pipeline complete.")

    # C. STANDARD VAE PIPELINE (Multi-VAE + Distillation + Trajectory)
    elif config.train.multivae:
        logger.log("Train multi vaes for each stage data (Standard Pipeline)")
        
        # Step 1: Train & Aggregate
        df_all, summary = train_multistage_vaes(
            tr_filter, te_filter,
            label_key=config.data.label_col,
            latent_dim=config.train.latent_dim,
            hidden_dims=config.train.hidden_dims,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            outdir=config.data.dir_out
        )
        
        # Step 2: Balance (Optional)
        if config.data.balance is not None:
            df_all = balance_latent_df(
                df_all,
                label_col=config.data.label_col,
                random_state=42,
            )
            
        # Step 3: Global UMAP (Raw VAE)
        logger.log("Generating Global UMAP for VAE Latents...")
        visualize_embeddings(
            df_all.filter(regex="^z\d+").values, 
            labels=df_all[config.data.label_col].values if config.data.label_col in df_all.columns else None,
            label_names=np.unique(df_all[config.data.label_col].values) if config.data.label_col in df_all.columns else None,
            save_path= os.path.join(config.data.dir_out, "global_umap.png")
        )
        
        # =========================================================
        # Step 4: Transformer Select Latent (Distillation)
        # =========================================================
        logger.log("Step 4: Refining Latent Space with Transformer (Distilling)...")
        
        # This distills the VAE features and logs the selected dimensions
        df_distilled = train_latent_transformer(
            latent_df=df_all,
            label_key=config.data.label_col,
            d_model=64,             
            nhead=4,
            num_layers=2,
            keep_ratio=0.5,         
            batch_size=32,
            epochs=100,
            outdir=os.path.join(config.data.dir_out, "step4_transformer_distill")
        )
        logger.log("Latent distillation complete.")
        
        # =========================================================
        # Step 5: Trajectory VAE (Optional)
        # =========================================================
        if args.time_sup:
            logger.log("\n**********************************")
            logger.log("Start Training Trajectory VAE to fix Order (1-2-3-4-5)")
        
            # Train the Trajectory VAE using the full VAE latent space
            df_trajectory = train_trajectory_vae(
                df_all,
                label_col="label",      
                latent_dim=2,          
                hidden_dims=[16, 8],   
                epochs=150,
                outdir=os.path.join(config.data.dir_out, "trajectory_model"),
                time_reg_weight=50.0    
            )
            
            # --- SAFE VISUALIZATION BLOCK ---
            traj_data = df_trajectory.filter(regex=r"^traj_z").values
            logger.log(f"Visualizing Trajectory: Found data shape {traj_data.shape}")
            
            if traj_data.shape[1] > 0:
                visualize_embeddings(
                    traj_data,
                    labels=df_trajectory[config.data.label_col].values,
                    label_names=np.unique(df_trajectory[config.data.label_col].values),
                    save_path=os.path.join(config.data.dir_out, "trajectory_umap.png")
                )
                logger.log("Trajectory visualization saved.")
            else:
                logger.log("[ERROR] No trajectory columns found (traj_z) to visualize. Skipping UMAP.")

    # D. SINGLE VAE
    else:
        logger.log("Train single vae using all stage data")
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
    
    logger.log("process complete")


def create_config():
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results_multistage",
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
            "hidden_dims":[256, 128, 64],
            "model_type": "vae",     # "vae", "transformer", or "transformer_on_latent"
            "top_k": 512,            # For transformer
            "d_model": 256,          # For transformer
            "lr": 1e-3
        },
        
    }
    return defaults  

if __name__ == "__main__":
    # Run the main pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    parser.add_argument("--time_sup", action='store_true')
    parser.add_argument("--model_type", type=str, default=None, help="vae, transformer, or transformer_on_latent")
    args = parser.parse_args()
    main(args)