"""Focused integrity and model-interface tests; not biological validation."""
import unittest,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Informer2020'))
from bin.model import ConditionalVAE
from train_cascaded import create_informer_model
from captum.attr import IntegratedGradients

class ReproductionTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'resources/input_checksums.json').exists(), 'Historical input data are not distributed')
    def test_inputs(self):
        r=ROOT/'resources';m=pd.read_csv(r/'sample_manifest.tsv',sep='\t',index_col=0)
        self.assertEqual(len(m),1138);self.assertTrue(m.index.is_unique)
        self.assertEqual(m['label'].value_counts().to_dict(),{'Stage_II':557,'Stage_III':219,'Stage_I':163,'normal_TCGA':110,'normal_GTEX':89})
        genes=(r/'model_genes.txt').read_text().splitlines();self.assertEqual(len(genes),4098)
        for entry in json.loads((r/'input_checksums.json').read_text()):
            path=r/entry['file'];self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
        for split in ['train','test']:
            df=pd.read_csv(r/f'model_input_{split}.tsv.gz',sep='\t',index_col=0)
            self.assertEqual(list(df.index),list(m.index[m.split==split]));self.assertEqual(list(df.columns),genes)

    def test_decoder_ig(self):
        torch.manual_seed(42)
        model=ConditionalVAE(12,latent_dim=4,hidden_dims=[8],n_classes=0).eval()
        z=torch.randn(2,4);baseline=torch.zeros_like(z)
        attr,delta=IntegratedGradients(lambda x:model.decode(x)[:,0]).attribute(z,baselines=baseline,n_steps=100,return_convergence_delta=True)
        self.assertEqual(tuple(attr.shape),(2,4));self.assertTrue(torch.isfinite(attr).all())
        self.assertLess(float(delta.abs().max()),0.02)

    def test_informer_forward(self):
        model=create_informer_model(32,64,4,2,torch.device('cpu')).eval()
        x=torch.randn(2,1,32);time=torch.zeros(2,1,5)
        with torch.no_grad():y=model(x,time,x,time)
        if isinstance(y,tuple):y=y[0]
        self.assertEqual(tuple(y.shape),(2,1,32));self.assertTrue(torch.isfinite(y).all())

if __name__=='__main__':unittest.main()
