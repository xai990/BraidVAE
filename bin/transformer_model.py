import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DistillingPruner(nn.Module):
    """
    The 'Trapezoid' mechanism for Genes.
    Instead of Convolution (which mixes genes), this uses Scoring (which selects genes).
    It reduces the sequence length, effectively 'distilling' the information.
    """
    def __init__(self, d_model, keep_ratio=0.5):
        super(DistillingPruner, self).__init__()
        self.keep_ratio = keep_ratio
        # Projects gene vector to a scalar score
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x, current_indices=None):
        # x: [Batch, Num_Genes, Dim]
        # current_indices: [Num_Genes] or None. Tracks original indices.

        # 1. Score genes (Learnable importance)
        scores = self.scorer(x) # [Batch, Num_Genes, 1]
        
        # 2. Consensus Score (Average across batch to find globally good genes)
        global_scores = scores.mean(dim=0).squeeze(-1) # [Num_Genes]
        
        # 3. Calculate how many to keep (k)
        num_genes = x.size(1)
        k = max(1, int(num_genes * self.keep_ratio))
        
        # 4. Select Top K indices
        _, top_k_indices = torch.topk(global_scores, k)
        top_k_indices, _ = torch.sort(top_k_indices) # Maintain relative order
        
        # 5. Prune the features (Drop the weak genes)
        x_distilled = x[:, top_k_indices, :]
        
        # 6. Update tracking of original indices
        if current_indices is not None:
            # If we already have indices from a previous layer, we select the subset from them
            new_indices = current_indices[top_k_indices]
        else:
            # If this is the first layer, the indices are just the top_k
            new_indices = top_k_indices
        
        return x_distilled, new_indices

class InformerEncoderLayer(nn.Module):
    """
    Standard Encoder Layer with Self-Attention and Feed Forward.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class GeneInformer(nn.Module):
    def __init__(self, num_genes, num_classes, d_model=128, nhead=4, num_encoder_layers=3, top_k=None, dim_feedforward=None, dropout=0.1, keep_ratio=0.5):
        super(GeneInformer, self).__init__()
        
        n_heads = nhead
        num_layers = num_encoder_layers
        
        # 1. Gene Embedding
        self.gene_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_genes, d_model))
        
        # 2. Build Layers Manually to manage the loop in forward()
        self.layers = nn.ModuleList()
        self.pruners = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(InformerEncoderLayer(d_model, n_heads, dropout))
            # Add Pruner to everyone except the last one
            if i < num_layers - 1:
                self.pruners.append(DistillingPruner(d_model, keep_ratio=keep_ratio))
            else:
                self.pruners.append(None) # Placeholder for last layer

        # 3. Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, return_indices=False):
        # x input: [Batch, Num_Genes]
        
        # Reshape & Embed
        x_in = x.unsqueeze(-1)
        x_emb = self.gene_embedding(x_in)
        
        # Add Position info
        if x_emb.size(1) <= self.pos_embedding.size(1):
             x_emb = x_emb + self.pos_embedding[:, :x_emb.size(1), :]
        
        x_curr = x_emb
        
        # Initialize indices tracking (0 to N-1)
        # This will keep track of which original gene column corresponds to the current vectors
        current_indices = torch.arange(x.size(1), device=x.device)

        # Loop through layers and prune
        for i, encoder_layer in enumerate(self.layers):
            # A. Run Attention/FFN
            x_curr = encoder_layer(x_curr)
            
            # B. Run Pruner (if it exists for this layer)
            pruner = self.pruners[i]
            if pruner is not None:
                x_curr, current_indices = pruner(x_curr, current_indices)

        # Final Pool
        x_pooled = self.global_pool(x_curr.transpose(1, 2)).squeeze(2)
        
        # Classify
        logits = self.classifier(x_pooled)
        
        if return_indices:
            # Return the indices of the genes that survived until the end
            # Use these indices to map back to your gene list (e.g. gene_list[idx])
            return logits, x_pooled, current_indices 
            
        return logits, x_pooled