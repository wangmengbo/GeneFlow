import os
import sys
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unet import RNAConditionedUNet # Assuming RNAConditionedUNet is in src.unet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Helper Residual Block for the encoder
class ResidualBlock(nn.Module):
    """Residual block with normalization and dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        # Skip connection with projection if dimensions don't match
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.main_branch(x) + self.skip(x)

# ======================================
# RNA Encoder with Cell Aggregation
# ======================================

class MultiCellRNAEncoder(nn.Module):
    """
    Enhanced encoder for multiple cells' RNA expression data with:
    1. Gene-aware attention mechanism
    2. Low-rank gene-gene relational modeling
    3. Improved cell aggregation using multi-head attention
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False,
                 dropout=0.1, use_gene_relations=True, relation_rank=50, num_aggregation_heads=4): # Added relation_rank, renamed num_heads
        super().__init__()
        self.input_dim = input_dim # Number of genes
        self.output_dim = output_dim
        self.concat_mask = concat_mask
        self.use_gene_relations = use_gene_relations
        self.relation_rank = relation_rank # K: the rank for factorization

        # Gene importance attention (applied after relational modeling)
        self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)

        if use_gene_relations:
            # This network part takes the raw gene expression x_flat [B*C, G]
            # and produces a cell-specific embedding.
            self.gene_relation_net_base = nn.Sequential(
                nn.Linear(input_dim, 256),  # Takes x_flat [B*C, G] as input
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            # This head predicts the parameters for the low-rank factor matrices U and V.
            # It needs to output 2 * num_genes * relation_rank parameters.
            self.gene_relation_factors_head = nn.Linear(256, 2 * input_dim * self.relation_rank)

        # Cell encoder layers with residual connections
        # This encoder processes each cell's (potentially relation-enhanced and attention-weighted) gene expression.
        cell_encoder_input_dim = input_dim
        if concat_mask:
            cell_encoder_input_dim = input_dim * 2 # If mask is concatenated to gene features

        prev_dim = cell_encoder_input_dim
        cell_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            cell_layers.append(nn.LayerNorm(prev_dim))
            cell_layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        self.cell_encoder = nn.Sequential(*cell_layers) # Outputs per-cell embeddings of size `prev_dim`

        # Multi-head attention for cell aggregation
        self.num_aggregation_heads = num_aggregation_heads # Renamed to avoid confusion
        # The input to this attention is the output of cell_encoder, which has `prev_dim` channels
        self.cell_aggregation_attention = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, prev_dim), # Or directly to num_aggregation_heads if simpler
            nn.SiLU(),
            nn.Linear(prev_dim, self.num_aggregation_heads)
        )
        
        # Head-specific projections for cell aggregation (optional, can make it more powerful)
        # Each head processes the `prev_dim` cell embedding
        self.aggregation_head_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, prev_dim), # Each head could learn a different transformation
                nn.SiLU()
            ) for _ in range(self.num_aggregation_heads)
        ])

        # Final encoding after cell aggregation
        # The input dimension here is `prev_dim` because we aggregate the `prev_dim` features from heads
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

        # Optional: feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

    def apply_gene_relations(self, x_input_genes):
        """Apply learned gene-gene relationships using low-rank factorization.
        x_input_genes shape: [batch_size, num_cells_in_patch, num_genes]
        """
        batch_size, num_cells_in_patch, num_genes = x_input_genes.shape
        # Flatten to process each cell's gene expression independently
        x_flat = x_input_genes.reshape(-1, num_genes)  # Shape: [B*C, G]

        # 1. Get cell-specific embedding from raw gene expression
        cell_embedding_for_relations = self.gene_relation_net_base(x_flat)  # [B*C, 256]

        # 2. Predict parameters for U and V factor matrices
        # Shape: [B*C, 2 * num_genes * relation_rank]
        relation_factors_params = self.gene_relation_factors_head(cell_embedding_for_relations)

        # 3. Reshape to get U [B*C, G, K] and V [B*C, K, G] matrices per cell
        U = relation_factors_params[:, :num_genes * self.relation_rank].view(
            batch_size * num_cells_in_patch, num_genes, self.relation_rank
        )
        V = relation_factors_params[:, num_genes * self.relation_rank:].view(
            batch_size * num_cells_in_patch, self.relation_rank, num_genes
        )

        # 4. Apply the transformation: x_transformed_contribution = (x_flat @ U) @ V
        x_flat_unsqueezed = x_flat.unsqueeze(1)  # [B*C, 1, G]
        temp = torch.bmm(x_flat_unsqueezed, U)   # [B*C, 1, K] (result of [B*C, 1, G] @ [B*C, G, K])
        x_transformed_flat = torch.bmm(temp, V).squeeze(1) # [B*C, G] (result of [B*C, 1, K] @ [B*C, K, G])

        # Reshape back to [B, C, G]
        x_transformed = x_transformed_flat.view(batch_size, num_cells_in_patch, num_genes)
        
        # Add the learned relational enhancement to the original expression
        return x_input_genes + 0.1 * x_transformed # Consider making 0.1 learnable or tunable

    def forward(self, x, mask=None, num_cells=None): # x is gene_expr [B, C_max, G]
        batch_size, max_cells_in_patch, num_genes = x.shape

        x_processed_relations = x
        if self.use_gene_relations:
            x_processed_relations = self.apply_gene_relations(x) # Output: [B, C_max, G]

        # Reshape to process all cells together for gene attention and initial encoding steps
        # Shape: [B * C_max, G]
        x_reshaped = x_processed_relations.reshape(batch_size * max_cells_in_patch, num_genes)

        # Apply gene attention (learned global importance for genes)
        # gene_attention is [G]
        gene_att_weights = F.softmax(self.gene_attention, dim=0)
        # x_weighted is [B*C_max, G]
        x_weighted = x_reshaped * gene_att_weights

        # Apply mask if provided (mask should correspond to x_reshaped if used here)
        if mask is not None and self.concat_mask:
            # Assuming mask is [B, C_max, G] and needs reshaping
            mask_reshaped = mask.reshape(batch_size * max_cells_in_patch, num_genes)
            x_weighted = torch.cat((x_weighted, mask_reshaped.to(x_weighted.dtype)), dim=1)
            # Note: cell_encoder_input_dim in __init__ must account for this doubling of features

        # Encode each cell's (modified) gene expression
        # cell_embeddings_flat is [B*C_max, cell_encoder_output_dim (prev_dim in __init__)]
        cell_embeddings_flat = self.cell_encoder(x_weighted)

        # Reshape back to [B, C_max, cell_encoder_output_dim] for aggregation
        cell_embeddings_batched = cell_embeddings_flat.reshape(batch_size, max_cells_in_patch, -1)

        # Create attention mask for valid cells (handling padding if num_cells is less than C_max)
        # This mask is for cell aggregation, not gene masking
        cell_agg_mask = torch.zeros(batch_size, max_cells_in_patch, 1, device=x.device)
        if num_cells is not None:
            for i, n_c in enumerate(num_cells): # num_cells is a list/tensor of actual cell counts per batch item
                cell_agg_mask[i, :n_c, :] = 1.0
        else: # If num_cells not provided, assume all cells in max_cells_in_patch are valid
            cell_agg_mask[:, :, :] = 1.0

        # --- Multi-head attention for cell aggregation ---
        # 1. Get attention logits for each head
        # cell_embeddings_batched: [B, C_max, D_cell_emb]
        # attention_logits: [B, C_max, num_aggregation_heads]
        attention_logits = self.cell_aggregation_attention(cell_embeddings_batched)

        # Apply mask to attention logits (before softmax)
        # Fill masked positions with a large negative number
        attention_logits = attention_logits.masked_fill(cell_agg_mask == 0, float('-inf'))

        # Transpose for softmax over cells per head: [B, num_aggregation_heads, C_max]
        attention_logits_transposed = attention_logits.permute(0, 2, 1)
        # cell_attention_weights: [B, num_aggregation_heads, C_max]
        cell_attention_weights = F.softmax(attention_logits_transposed, dim=2)

        # 2. Apply head-specific projections and aggregate
        aggregated_head_outputs = []
        for h in range(self.num_aggregation_heads):
            # Projected cell embeddings for this head: [B, C_max, D_cell_emb]
            projected_embeddings = self.aggregation_head_projections[h](cell_embeddings_batched)
            
            # Weights for this head: [B, 1, C_max] (unsqueeze for broadcasting)
            current_head_weights = cell_attention_weights[:, h, :].unsqueeze(1)
            
            # Weighted sum: [B, 1, C_max] @ [B, C_max, D_cell_emb] -> [B, 1, D_cell_emb]
            weighted_sum = torch.bmm(current_head_weights, projected_embeddings)
            aggregated_head_outputs.append(weighted_sum.squeeze(1)) # Squeeze to [B, D_cell_emb]

        # Combine head outputs - e.g., by averaging or concatenating
        # Averaging to keep dimension `D_cell_emb` (which is `prev_dim` from __init__)
        if self.num_aggregation_heads > 0:
            aggregated_features = torch.stack(aggregated_head_outputs, dim=1).mean(dim=1) # [B, D_cell_emb]
        else: # Fallback if no heads (should not happen with num_aggregation_heads > 0)
            aggregated_features = cell_embeddings_batched.mean(dim=1) # Simple averaging if no attention heads


        # Final encoding layer
        final_embeddings = self.final_encoder(aggregated_features) # Input [B, D_cell_emb], Output [B, output_dim]

        # Apply feature gating
        gates = self.feature_gate(final_embeddings)
        gated_embeddings = final_embeddings * gates

        return gated_embeddings

    def get_gene_importance(self):
        """Return the learned importance of each gene (global attention)"""
        return F.softmax(self.gene_attention, dim=0)


# ======================================
# Multi-Cell RNA to H&E Image Model
# ======================================

class MultiCellRNAtoHnEModel(nn.Module):
    """
    Model for generating H&E patch images from multiple cells' RNA expression data
    using advanced flow matching techniques.
    """
    def __init__(
        self,
        rna_dim, # This is input_dim (num_genes) for MultiCellRNAEncoder
        img_channels=3,
        img_size=64,
        model_channels=128, # UNet model_channels
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        use_checkpoint=False,
        # UNet attention heads, not to be confused with cell aggregation heads
        num_heads=2,
        num_head_channels=16,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        concat_mask=False,
        # Parameters for MultiCellRNAEncoder
        encoder_hidden_dims=[512, 256],
        encoder_output_dim_multiplier=4, # Multiplies model_channels for rna_embed_dim
        use_gene_relations=True,
        relation_rank=50,
        num_aggregation_heads=4
    ):
        super().__init__()

        self.rna_dim = rna_dim
        self.img_channels = img_channels
        self.img_size = img_size

        rna_encoder_output_dim = model_channels * encoder_output_dim_multiplier

        # Multi-cell RNA expression encoder
        self.rna_encoder = MultiCellRNAEncoder(
            input_dim=rna_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=rna_encoder_output_dim, # This will be rna_embed_dim for UNet
            concat_mask=concat_mask,
            dropout=dropout,
            use_gene_relations=use_gene_relations,
            relation_rank=relation_rank,
            num_aggregation_heads=num_aggregation_heads
        )

        # UNet model for flow matching
        self.unet = RNAConditionedUNet(
            in_channels=img_channels,
            model_channels=model_channels,
            out_channels=img_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads, # Pass UNet specific head count
            num_head_channels=num_head_channels, # Pass UNet specific head channels
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            rna_embed_dim=rna_encoder_output_dim, # Matches output of rna_encoder
        )

    def forward(self, x, t, gene_expr, num_cells=None, gene_mask=None):
        """
        Forward pass for the Multi-Cell RNA to H&E model

        Args:
            x: Input image tensor [B, C_img, H, W]
            t: Timestep tensor [B]
            gene_expr: RNA expression tensor [B, C_max_cells, G_genes]
            num_cells: Actual number of cells per patch [B], used for masking in aggregation
            gene_mask: Optional mask for missing genes [B, C_max_cells, G_genes]

        Returns:
            Predicted velocity field for the flow matching model
        """
        # Encode RNA expression for multiple cells
        rna_embedding = self.rna_encoder(gene_expr, mask=gene_mask, num_cells=num_cells)

        # Get vector field from UNet model
        return self.unet(x, t, extra={"rna_embedding": rna_embedding})


# ======================================
# Utility to convert batch data (if needed, e.g. from rectified_main.py)
# ======================================
# The prepare_multicell_batch function from your original multi_model.py
# seems fine for preparing data from the DataLoader.

def prepare_multicell_batch(batch, device):
    """
    Prepare a batch from PatchImageGeneDataset for input to MultiCellRNAtoHnEModel
    (Copied from your provided code, ensure it's what you need)
    Args:
        batch: Dictionary with keys: 'patch_id', 'cell_ids', 'gene_expr', 'image', 'num_cells'
        device: Target device for tensors
    Returns:
        Dictionary with model-ready tensors
    """
    images = batch['image'].to(device)
    num_cells = batch['num_cells'] # This should be a list or tensor of actual cell counts

    gene_expr_tensor = batch['gene_expr'] # Assuming this is already padded [B, C_max, G]
    if not isinstance(gene_expr_tensor, torch.Tensor):
        # This case might occur if collate_fn isn't padding correctly,
        # but patch_collate_fn from dataset.py should handle it.
        logger.warning("gene_expr is not a tensor; attempting to convert/pad. Ensure collate_fn is used.")
        # Basic padding if it's a list of tensors (example, adapt as needed)
        if isinstance(gene_expr_tensor, list):
            # This requires knowing max_cells and gene_dim beforehand or from the first element
            # For safety, this part should ideally be handled by a robust collate_fn
            max_cells_in_batch = max(s.shape[0] for s in gene_expr_tensor)
            gene_dim = gene_expr_tensor[0].shape[1]
            padded_gene_expr = torch.zeros(len(gene_expr_tensor), max_cells_in_batch, gene_dim, dtype=gene_expr_tensor[0].dtype, device=device)
            for i, expr in enumerate(gene_expr_tensor):
                padded_gene_expr[i, :expr.shape[0]] = expr.to(device)
            gene_expr_tensor = padded_gene_expr
    else:
        gene_expr_tensor = gene_expr_tensor.to(device)

    # Validate num_cells against the tensor's second dimension if it's a list/tensor of counts
    # The `num_cells` from patch_collate_fn is torch.tensor(num_cells), which is good.
    # It will be used by MultiCellRNAEncoder for masking during aggregation.

    return {
        'image': images,
        'gene_expr': gene_expr_tensor, # Should be [B, C_max, G]
        'num_cells': num_cells # Should be [B], indicating actual cells per item
    }


# import os
# import sys
# import torch
# import logging
# import torch.nn as nn
# import torch.nn.functional as F

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.unet import RNAConditionedUNet

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

# # ======================================
# # RNA Encoder with Cell Aggregation
# # ======================================

# # class MultiCellRNAEncoder(nn.Module):
# #     """
# #     Encoder for multiple cells' RNA expression data with attention-based aggregation.
# #     This encoder can handle variable numbers of cells per patch.
# #     """
# #     def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False):
# #         super().__init__()
# #         # Gene attention weights
# #         self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)
# #         self.concat_mask = concat_mask
        
# #         # Cell encoder layers
# #         cell_layers = []
# #         if concat_mask:
# #             first_layer_input_dim = input_dim * 2
# #             prev_dim = first_layer_input_dim
# #         else:
# #             prev_dim = input_dim
        
# #         for hidden_dim in hidden_dims:
# #             cell_layers.append(nn.Linear(prev_dim, hidden_dim))
# #             cell_layers.append(nn.LayerNorm(hidden_dim))
# #             cell_layers.append(nn.SiLU())
# #             prev_dim = hidden_dim
            
# #         # Cell embedding layer - produces per-cell embeddings
# #         self.cell_encoder = nn.Sequential(*cell_layers)
        
# #         # Cell attention for weighted aggregation
# #         self.cell_attention = nn.Sequential(
# #             nn.Linear(prev_dim, 128),
# #             nn.SiLU(),
# #             nn.Linear(128, 1)
# #         )
        
# #         # Final encoding after cell aggregation
# #         self.final_encoder = nn.Sequential(
# #             nn.Linear(prev_dim, output_dim),
# #             nn.LayerNorm(output_dim)
# #         )
        
# #     def forward(self, x, mask=None, num_cells=None):
# #         """
# #         Forward pass for multiple cells' RNA expression data
        
# #         Args:
# #             x: RNA expression tensor [B, C, G] where:
# #                B = batch size
# #                C = max number of cells per patch
# #                G = number of genes
# #             mask: Optional mask for missing genes [B, C, G]
# #             num_cells: Number of cells per patch [B]
            
# #         Returns:
# #             Aggregated RNA embeddings [B, output_dim]
# #         """
# #         batch_size, max_cells, num_genes = x.shape
        
# #         # Reshape to process all cells together
# #         x_reshaped = x.reshape(batch_size * max_cells, num_genes)
        
# #         # Apply gene attention
# #         attention = F.softmax(self.gene_attention, dim=0)
# #         x_weighted = x_reshaped * attention
        
# #         # Apply mask if provided
# #         if mask is not None and self.concat_mask:
# #             mask_reshaped = mask.reshape(batch_size * max_cells, num_genes)
# #             x_weighted = torch.cat((x_weighted, mask_reshaped.to(x_weighted.dtype)), dim=1)
        
# #         # Encode each cell
# #         cell_embeddings = self.cell_encoder(x_weighted)
        
# #         # Reshape back to [B, C, H]
# #         cell_embeddings = cell_embeddings.reshape(batch_size, max_cells, -1)
        
# #         # Create attention mask for valid cells
# #         if num_cells is not None:
# #             # Create a mask where 1s represent valid cells
# #             attention_mask = torch.zeros(batch_size, max_cells, 1, device=x.device)
# #             for i, num in enumerate(num_cells):
# #                 attention_mask[i, :num, 0] = 1.0
# #         else:
# #             # If num_cells is not provided, assume all cells are valid
# #             attention_mask = torch.ones(batch_size, max_cells, 1, device=x.device)
        
# #         # Calculate attention scores for each cell
# #         attention_logits = self.cell_attention(cell_embeddings)
        
# #         # Apply mask to attention logits
# #         attention_logits = attention_logits * attention_mask - 1e9 * (1 - attention_mask)
        
# #         # Compute attention weights
# #         cell_attention_weights = F.softmax(attention_logits, dim=1)
        
# #         # Apply attention weights to get weighted sum of cell embeddings
# #         weighted_embeddings = (cell_embeddings * cell_attention_weights).sum(dim=1)
        
# #         # Final encoding
# #         final_embeddings = self.final_encoder(weighted_embeddings)
        
# #         return final_embeddings
        
# #     def get_gene_importance(self):
# #         """Return the learned importance of each gene"""
# #         return F.softmax(self.gene_attention, dim=0)

# class MultiCellRNAEncoder(nn.Module):
#     """
#     Enhanced encoder for multiple cells' RNA expression data with:
#     1. Gene-aware attention mechanism
#     2. Gene-gene relational modeling
#     3. Improved cell aggregation
#     """
#     def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False, 
#                  dropout=0.1, use_gene_relations=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.concat_mask = concat_mask
#         self.use_gene_relations = use_gene_relations
        
#         # Gene importance attention with prior biological knowledge
#         # This learns which genes are most important for cell morphology
#         self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)
        
#         # Gene-gene relational modeling
#         if use_gene_relations:
#             self.gene_relation_net = nn.Sequential(
#                 nn.Linear(input_dim, 256),
#                 nn.LayerNorm(256),
#                 nn.SiLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(256, input_dim * input_dim),
#             )
        
#         # Cell encoder layers with residual connections
#         cell_layers = []
#         if concat_mask:
#             first_layer_input_dim = input_dim * 2
#             prev_dim = first_layer_input_dim
#         else:
#             prev_dim = input_dim
        
#         for i, hidden_dim in enumerate(hidden_dims):
#             # Add residual block for each hidden layer
#             cell_layers.append(nn.LayerNorm(prev_dim))
#             cell_layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
#             prev_dim = hidden_dim
            
#         # Cell embedding layer - produces per-cell embeddings
#         self.cell_encoder = nn.Sequential(*cell_layers)
        
#         # Multi-head attention for cell aggregation
#         self.num_heads = 4
#         self.head_dim = prev_dim // self.num_heads
#         self.cell_attention = nn.Sequential(
#             nn.LayerNorm(prev_dim),
#             nn.Linear(prev_dim, prev_dim),
#             nn.SiLU(),
#             nn.Linear(prev_dim, self.num_heads)
#         )
        
#         # Final encoding after cell aggregation with normalization
#         self.final_encoder = nn.Sequential(
#             nn.LayerNorm(prev_dim),
#             nn.Linear(prev_dim, output_dim),
#             nn.Dropout(dropout),
#             nn.LayerNorm(output_dim)
#         )
        
#         # Optional: feature gating mechanism for dynamic feature selection
#         self.feature_gate = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.Sigmoid()
#         )
        
#     def apply_gene_relations(self, x):
#         """Apply learned gene-gene relationships to expression data"""
#         batch_size, num_cells, num_genes = x.shape
        
#         # Reshape to process each cell separately
#         x_flat = x.reshape(-1, num_genes)  # [batch_size * num_cells, num_genes]
        
#         # Predict gene-gene relationship matrix (can be interpreted as pathway effects)
#         rel_matrix = self.gene_relation_net(x_flat)
#         rel_matrix = rel_matrix.view(-1, num_genes, num_genes)
        
#         # Apply relationship matrix to enhance gene expressions
#         enhanced_expr = torch.bmm(x_flat.unsqueeze(1), rel_matrix).squeeze(1)
#         enhanced_expr = enhanced_expr.view(batch_size, num_cells, num_genes)
        
#         # Residual connection to preserve original expression information
#         return x + 0.1 * enhanced_expr
        
#     def forward(self, x, mask=None, num_cells=None):
#         """
#         Forward pass with enhanced biological awareness
        
#         Args:
#             x: RNA expression tensor [B, C, G] where:
#                B = batch size
#                C = max number of cells per patch
#                G = number of genes
#             mask: Optional mask for missing genes [B, C, G]
#             num_cells: Number of cells per patch [B]
            
#         Returns:
#             Aggregated RNA embeddings [B, output_dim]
#         """
#         batch_size, max_cells, num_genes = x.shape
        
#         # Apply gene-gene relational modeling if enabled
#         if self.use_gene_relations:
#             x = self.apply_gene_relations(x)
        
#         # Reshape to process all cells together
#         x_reshaped = x.reshape(batch_size * max_cells, num_genes)
        
#         # Apply gene attention with softmax for proper normalization
#         attention = F.softmax(self.gene_attention, dim=0)
#         x_weighted = x_reshaped * attention
        
#         # Apply mask if provided
#         if mask is not None and self.concat_mask:
#             mask_reshaped = mask.reshape(batch_size * max_cells, num_genes)
#             x_weighted = torch.cat((x_weighted, mask_reshaped.to(x_weighted.dtype)), dim=1)
        
#         # Encode each cell with our enhanced encoder
#         cell_embeddings = self.cell_encoder(x_weighted)
        
#         # Reshape back to [B, C, H]
#         cell_embeddings = cell_embeddings.reshape(batch_size, max_cells, -1)
        
#         # Create attention mask for valid cells
#         if num_cells is not None:
#             # Create a mask where 1s represent valid cells
#             attention_mask = torch.zeros(batch_size, max_cells, 1, device=x.device)
#             for i, num in enumerate(num_cells):
#                 attention_mask[i, :num, 0] = 1.0
#         else:
#             # If num_cells is not provided, assume all cells are valid
#             attention_mask = torch.ones(batch_size, max_cells, 1, device=x.device)
        
#         # Multi-head attention for cell aggregation
#         attention_logits = self.cell_attention(cell_embeddings)  # [B, C, num_heads]
#         attention_logits = attention_logits.permute(0, 2, 1)  # [B, num_heads, C]
        
#         # Apply mask to attention logits (per head)
#         attention_mask = attention_mask.squeeze(-1).unsqueeze(1)  # [B, 1, C]
#         attention_logits = attention_logits * attention_mask - 1e9 * (1 - attention_mask)
        
#         # Compute attention weights (per head)
#         cell_attention_weights = F.softmax(attention_logits, dim=2)  # [B, num_heads, C]
        
#         # We need to do the aggregation for each batch and head separately
        
#         # Initialize output tensor for collected head outputs
#         head_output_list = []
        
#         # Process each head separately to avoid dimension mismatch
#         for h in range(self.num_heads):
#             # Get attention weights for this head [B, C]
#             head_weights = cell_attention_weights[:, h, :]
            
#             # Apply attention weights to cell embeddings for this head
#             # [B, C, 1] * [B, C, H] -> [B, C, H] -> sum over C -> [B, H]
#             weighted_cells = (head_weights.unsqueeze(-1) * cell_embeddings).sum(dim=1)
#             head_output_list.append(weighted_cells)
        
#         # Concatenate or average the head outputs
#         if len(head_output_list) > 1:
#             # Option 1: Concatenate heads (if you want to preserve information from each head)
#             # weighted_embeddings = torch.cat(head_output_list, dim=1)
            
#             # Option 2: Average heads (if you want to maintain the same dimension)
#             weighted_embeddings = torch.stack(head_output_list, dim=1).mean(dim=1)
#         else:
#             weighted_embeddings = head_output_list[0]
        
#         # Final encoding
#         final_embeddings = self.final_encoder(weighted_embeddings)
        
#         # Apply feature gating for dynamic feature selection
#         gates = self.feature_gate(final_embeddings)
#         gated_embeddings = final_embeddings * gates
        
#         return gated_embeddings
        
#     def get_gene_importance(self):
#         """Return the learned importance of each gene"""
#         return F.softmax(self.gene_attention, dim=0)

# # Helper Residual Block for the encoder
# class ResidualBlock(nn.Module):
#     """Residual block with normalization and dropout"""
#     def __init__(self, in_dim, out_dim, dropout=0.1):
#         super().__init__()
#         self.main_branch = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#             nn.Dropout(dropout)
#         )
        
#         # Skip connection with projection if dimensions don't match
#         self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
#     def forward(self, x):
#         return self.main_branch(x) + self.skip(x)

# # ======================================
# # Multi-Cell RNA to H&E Image Model
# # ======================================

# class MultiCellRNAtoHnEModel(nn.Module):
#     """
#     Model for generating H&E patch images from multiple cells' RNA expression data
#     using advanced flow matching techniques.
#     """
#     def __init__(
#         self,
#         rna_dim,
#         img_channels=3,
#         img_size=64,
#         model_channels=128,
#         num_res_blocks=2,
#         attention_resolutions=[16],
#         dropout=0.1,
#         channel_mult=(1, 2, 2, 2),
#         use_checkpoint=False,
#         num_heads=2,
#         num_head_channels=16,
#         use_scale_shift_norm=True,
#         resblock_updown=True,
#         use_new_attention_order=True,
#         concat_mask=False,
#     ):
#         super().__init__()
        
#         self.rna_dim = rna_dim
#         self.img_channels = img_channels
#         self.img_size = img_size

#         # Multi-cell RNA expression encoder
#         self.rna_encoder = MultiCellRNAEncoder(
#             input_dim=rna_dim,
#             hidden_dims=[512, 256],
#             output_dim=model_channels * 4,  # Match time_embed_dim
#             concat_mask=concat_mask,
#         )
        
#         # UNet model for flow matching - same as single cell model
#         self.unet = RNAConditionedUNet(
#             in_channels=img_channels,
#             model_channels=model_channels,
#             out_channels=img_channels,
#             num_res_blocks=num_res_blocks,
#             attention_resolutions=attention_resolutions,
#             dropout=dropout,
#             channel_mult=channel_mult,
#             use_checkpoint=use_checkpoint,
#             num_heads=num_heads,
#             num_head_channels=num_head_channels,
#             use_scale_shift_norm=use_scale_shift_norm,
#             resblock_updown=resblock_updown,
#             use_new_attention_order=use_new_attention_order,
#             rna_embed_dim=model_channels * 4,
#         )
        
#     def forward(self, x, t, gene_expr, num_cells=None, gene_mask=None):
#         """
#         Forward pass for the Multi-Cell RNA to H&E model
        
#         Args:
#             x: Input image tensor [B, C, H, W]
#             t: Timestep tensor [B]
#             gene_expr: RNA expression tensor [B, num_cells, rna_dim]
#             num_cells: Number of cells per patch [B]
#             gene_mask: Optional mask for missing genes [B, num_cells, rna_dim]
            
#         Returns:
#             Predicted velocity field for the flow matching model
#         """
#         # Encode RNA expression for multiple cells
#         rna_embedding = self.rna_encoder(gene_expr, mask=gene_mask, num_cells=num_cells)
        
#         # Get vector field from UNet model
#         return self.unet(x, t, extra={"rna_embedding": rna_embedding})


# # ======================================
# # Utility to convert batch data from PatchImageGeneDataset for model input
# # ======================================

# def prepare_multicell_batch(batch, device):
#     """
#     Prepare a batch from PatchImageGeneDataset for input to MultiCellRNAtoHnEModel
    
#     Args:
#         batch: Dictionary with keys: 'patch_id', 'cell_ids', 'gene_expr', 'image', 'num_cells'
#         device: Target device for tensors
        
#     Returns:
#         Dictionary with model-ready tensors
#     """
#     images = batch['image'].to(device)
#     num_cells = batch['num_cells']
    
#     # Check if gene_expr is already a padded tensor (from patch_collate_fn)
#     gene_expr = batch['gene_expr']
#     if isinstance(gene_expr, torch.Tensor):
#         # gene_expr is already padded [batch_size, max_cells, gene_dim]
#         padded_gene_expr = gene_expr.to(device)
#         batch_size, max_cells, gene_dim = padded_gene_expr.shape
        
#         # Validate num_cells
#         for i, n_cells in enumerate(num_cells):
#             if n_cells > max_cells:
#                 logger.error(f"Sample {i}: num_cells={n_cells} exceeds max_cells={max_cells}")
#                 raise ValueError(f"Sample {i}: num_cells={n_cells} exceeds max_cells={max_cells}")
#     else:
#         # Fallback: pad gene_expr manually (for compatibility with older code)
#         logger.warning("gene_expr is a list; padding manually")
#         batch_size = len(num_cells)
#         max_cells = max(num_cells)
#         gene_dim = gene_expr[0].shape[1]
#         padded_gene_expr = torch.zeros(batch_size, max_cells, gene_dim, device=device)
        
#         for i, sample_gene_expr in enumerate(gene_expr):
#             cells_count = sample_gene_expr.shape[0]
#             if cells_count != num_cells[i]:
#                 logger.error(f"Mismatch in sample {i}: num_cells={num_cells[i]}, but gene_expr has {cells_count} cells")
#                 raise ValueError(f"Mismatch in sample {i}: num_cells={num_cells[i]}, but gene_expr has {cells_count} cells")
#             padded_gene_expr[i, :cells_count] = sample_gene_expr.to(device)
    
#     return {
#         'image': images,
#         'gene_expr': padded_gene_expr,
#         'num_cells': num_cells
#     }