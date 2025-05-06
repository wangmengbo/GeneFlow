import os
import sys
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unet import RNAConditionedUNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# RNA Encoder
# ======================================

# class RNAEncoder(nn.Module):
#     """
#     Encoder for RNA expression data that produces embeddings for conditioning the UNet.
#     """
#     def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False):
#         super().__init__()
#         # Add attention weights for genes
#         self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)
#         # Rest of encoder remains the same
#         layers = []

#         if concat_mask:
#             self.concat_mask = True
#             first_layer_input_dim = input_dim * 2
#             prev_dim = first_layer_input_dim
#         else:
#             self.concat_mask = False
#             prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.LayerNorm(hidden_dim))
#             layers.append(nn.SiLU())
#             prev_dim = hidden_dim
            
#         layers.append(nn.Linear(prev_dim, output_dim))
#         self.encoder = nn.Sequential(*layers)
        
#     def forward(self, x, mask=None):
#         # Apply attention to genes
#         attention = F.softmax(self.gene_attention, dim=0)
#         x_weighted = x * attention
#         if mask is not None and self.concat_mask:
#             x_weighted = torch.cat((x_weighted, mask.to(x_weighted.dtype)), dim=1)
#         return self.encoder(x_weighted)
        
#     def get_gene_importance(self):
#         """Return the learned importance of each gene"""
#         return F.softmax(self.gene_attention, dim=0)

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

class RNAEncoder(nn.Module):
    """
    Enhanced encoder for RNA expression data with:
    1. Gene-aware attention mechanism
    2. Gene-gene relational modeling
    3. Multi-head attention for feature extraction
    4. Feature gating for dynamic selection
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False,
                 dropout=0.1, use_gene_relations=True, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat_mask = concat_mask
        self.use_gene_relations = use_gene_relations
        self.num_heads = num_heads
        
        # Gene importance attention
        self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)
        
        # Gene-gene relational modeling (optional)
        if use_gene_relations:
            self.gene_relation_net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(256, input_dim * input_dim),
            )
        
        # Encoder layers with residual connections
        layers = []
        if concat_mask:
            first_layer_input_dim = input_dim * 2
            prev_dim = first_layer_input_dim
        else:
            prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Add residual block for each hidden layer
            layers.append(nn.LayerNorm(prev_dim))
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Multi-head attention for feature extraction (similar to MultiCellRNAEncoder)
        self.feature_attention = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, prev_dim),
            nn.SiLU(),
            nn.Linear(prev_dim, self.num_heads)
        )
        
        # Head-specific feature extraction
        self.head_dim = prev_dim
        self.head_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, prev_dim),
                nn.SiLU()
            ) for _ in range(self.num_heads)
        ])
        
        # Final integration layer
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
        # Feature gating mechanism for dynamic feature selection
        self.feature_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
    def apply_gene_relations(self, x):
        """Apply learned gene-gene relationships to expression data"""
        batch_size, num_genes = x.shape
        
        # Predict gene-gene relationship matrix
        rel_matrix = self.gene_relation_net(x)
        rel_matrix = rel_matrix.view(batch_size, num_genes, num_genes)
        
        # Apply relationship matrix to enhance gene expressions
        enhanced_expr = torch.bmm(x.unsqueeze(1), rel_matrix).squeeze(1)
        
        # Residual connection to preserve original expression information
        return x + 0.1 * enhanced_expr
        
    def forward(self, x, mask=None):
        """
        Forward pass with enhanced biological awareness and multi-head attention
        
        Args:
            x: RNA expression tensor [B, G] where:
               B = batch size
               G = number of genes
            mask: Optional mask for missing genes [B, G]
            
        Returns:
            RNA embeddings [B, output_dim]
        """
        batch_size, num_genes = x.shape
        
        # Apply gene-gene relational modeling if enabled
        if self.use_gene_relations:
            x = self.apply_gene_relations(x)
        
        # Apply gene attention with softmax
        attention = F.softmax(self.gene_attention, dim=0)
        x_weighted = x * attention
        
        # Apply mask if provided
        if mask is not None and self.concat_mask:
            x_weighted = torch.cat((x_weighted, mask.to(x_weighted.dtype)), dim=1)
        
        # Encode RNA expression
        embeddings = self.encoder(x_weighted)  # [B, H]
        
        # Calculate attention scores for each head
        attention_logits = self.feature_attention(embeddings)  # [B, num_heads]
        
        # Get attention weights for each head
        head_attention_weights = F.softmax(attention_logits, dim=1)  # [B, num_heads]
        
        # Process through each head
        head_outputs = []
        for h in range(self.num_heads):
            # Apply projection for this head
            head_features = self.head_projections[h](embeddings)  # [B, H]
            head_outputs.append(head_features)
        
        # Stack head outputs
        stacked_outputs = torch.stack(head_outputs, dim=1)  # [B, num_heads, H]
        
        # Apply attention weights to combine head outputs
        head_attention_weights = head_attention_weights.unsqueeze(-1)  # [B, num_heads, 1]
        weighted_outputs = stacked_outputs * head_attention_weights  # [B, num_heads, H]
        
        # Sum over heads
        aggregated_features = weighted_outputs.sum(dim=1)  # [B, H]
        
        # Final encoding
        final_embeddings = self.final_encoder(aggregated_features)
        
        # Apply feature gating for dynamic feature selection
        gates = self.feature_gate(final_embeddings)
        gated_embeddings = final_embeddings * gates
        
        return gated_embeddings
    
    def get_gene_importance(self):
        """Return the learned importance of each gene"""
        return F.softmax(self.gene_attention, dim=0)
    
# ======================================
# RNA to H&E Cell Image Model
# ======================================

class RNAtoHnEModel(nn.Module):
    """
    Complete model for generating H&E cell images from RNA expression data
    using advanced flow matching techniques.
    """
    def __init__(
        self,
        rna_dim,
        img_channels=3,
        img_size=64,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        use_checkpoint=False,
        num_heads=2,
        num_head_channels=16,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        concat_mask=False,
    ):
        super().__init__()
        
        self.rna_dim = rna_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # RNA expression encoder
        self.rna_encoder = RNAEncoder(
            input_dim=rna_dim,
            hidden_dims=[512, 256],
            output_dim=model_channels * 4,  # Match time_embed_dim
            concat_mask=concat_mask,
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
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            rna_embed_dim=model_channels * 4,
        )
        
    def forward(self, x, t, gene_expr, gene_mask=None):
        """
        Forward pass for the RNA to H&E model
        
        Args:
            x: Input image tensor [B, C, H, W]
            t: Timestep tensor [B]
            rna_expr: RNA expression tensor [B, rna_dim]
            
        Returns:
            Predicted velocity field for the flow matching model
        """
        # Encode RNA expression
        rna_embedding = self.rna_encoder(gene_expr, mask=gene_mask)
        
        # Get vector field from UNet model
        return self.unet(x, t, extra={"rna_embedding": rna_embedding})