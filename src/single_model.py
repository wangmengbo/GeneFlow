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
    Enhanced encoder for RNA expression data with ablation capabilities.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, concat_mask=False,
                 dropout=0.1, use_gene_relations=True, num_heads=4, relation_rank=50,
                 # Ablation flags
                 use_gene_attention=True,
                 use_multi_head_attention=True, 
                 use_feature_gating=True,
                 use_residual_blocks=True,
                 use_layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat_mask = concat_mask
        self.use_gene_relations = use_gene_relations
        self.num_heads = num_heads
        self.relation_rank = relation_rank
        
        # Ablation flags
        self.use_gene_attention = use_gene_attention
        self.use_multi_head_attention = use_multi_head_attention
        self.use_feature_gating = use_feature_gating
        self.use_residual_blocks = use_residual_blocks
        self.use_layer_norm = use_layer_norm

        # Gene importance attention (ablatable)
        if self.use_gene_attention:
            self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)

        # Gene relations network (existing ablation via use_gene_relations)
        if use_gene_relations:
            self.gene_relation_net_base = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256) if self.use_layer_norm else nn.Identity(),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.gene_relation_factors_head = nn.Linear(256, 2 * input_dim * self.relation_rank)

        # Encoder layers with optional residual connections and layer norm
        layers = []
        current_encoder_input_dim = input_dim
        if concat_mask:
            current_encoder_input_dim = input_dim * 2
        
        prev_dim = current_encoder_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(prev_dim))
            
            if self.use_residual_blocks:
                layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            else:
                # Simple linear layers without residual connections
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Multi-head attention for feature extraction (ablatable)
        if self.use_multi_head_attention:
            self.feature_attention = nn.Sequential(
                nn.LayerNorm(prev_dim) if self.use_layer_norm else nn.Identity(),
                nn.Linear(prev_dim, prev_dim),
                nn.SiLU(),
                nn.Linear(prev_dim, self.num_heads)
            )

            self.head_projections = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(prev_dim) if self.use_layer_norm else nn.Identity(),
                    nn.Linear(prev_dim, prev_dim),
                    nn.SiLU()
                ) for _ in range(self.num_heads)
            ])

        # Final integration layer
        final_layers = []
        if self.use_layer_norm:
            final_layers.append(nn.LayerNorm(prev_dim))
        final_layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout)
        ])
        if self.use_layer_norm:
            final_layers.append(nn.LayerNorm(output_dim))
        
        self.final_encoder = nn.Sequential(*final_layers)

        # Feature gating mechanism (ablatable)
        if self.use_feature_gating:
            self.feature_gate = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid()
            )

    def apply_gene_relations(self, x):
        """Apply learned gene-gene relationships using low-rank factorization."""
        batch_size, num_genes = x.shape

        cell_embedding_for_relations = self.gene_relation_net_base(x)
        relation_factors_params = self.gene_relation_factors_head(cell_embedding_for_relations)

        U = relation_factors_params[:, :num_genes * self.relation_rank].view(
            batch_size, num_genes, self.relation_rank
        )
        V = relation_factors_params[:, num_genes * self.relation_rank:].view(
            batch_size, self.relation_rank, num_genes
        )

        x_unsqueezed = x.unsqueeze(1)
        temp = torch.bmm(x_unsqueezed, U)
        x_transformed_by_uv = torch.bmm(temp, V).squeeze(1)

        return x + 0.1 * x_transformed_by_uv

    def forward(self, x, mask=None):
        batch_size, num_genes = x.shape

        x_processed = x
        if self.use_gene_relations:
            x_processed = self.apply_gene_relations(x_processed)

        # Apply gene attention (ablatable)
        if self.use_gene_attention:
            attention = F.softmax(self.gene_attention, dim=0)
            x_weighted = x_processed * attention
        else:
            x_weighted = x_processed

        if mask is not None and self.concat_mask:
            x_weighted = torch.cat((x_weighted, mask.to(x_weighted.dtype)), dim=1)

        embeddings = self.encoder(x_weighted)

        # Multi-head attention processing (ablatable)
        if self.use_multi_head_attention:
            attention_logits = self.feature_attention(embeddings)
            head_attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)

            head_outputs = torch.stack([proj(embeddings) for proj in self.head_projections], dim=1)
            
            weighted_outputs = head_outputs * head_attention_weights
            aggregated_features = weighted_outputs.sum(dim=1)
        else:
            # Simple processing without multi-head attention
            aggregated_features = embeddings

        final_embeddings = self.final_encoder(aggregated_features)
        
        # Feature gating (ablatable)
        if self.use_feature_gating:
            gates = self.feature_gate(final_embeddings)
            gated_embeddings = final_embeddings * gates
        else:
            gated_embeddings = final_embeddings

        return gated_embeddings

    def get_gene_importance(self):
        if self.use_gene_attention:
            return F.softmax(self.gene_attention, dim=0)
        else:
            return torch.ones(self.input_dim) / self.input_dim  # Uniform importance if no attention

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
        relation_rank=50,
        # Ablation parameters
        use_gene_attention=True,
        use_multi_head_attention=True,
        use_feature_gating=True,
        use_residual_blocks=True,
        use_layer_norm=True,
        use_gene_relations=True,
    ):
        super().__init__()
        
        self.rna_dim = rna_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # RNA expression encoder with ablation support
        self.rna_encoder = RNAEncoder(
            input_dim=rna_dim,
            hidden_dims=[512, 256],
            output_dim=model_channels * 4,  # Match time_embed_dim
            concat_mask=concat_mask,
            relation_rank=relation_rank,
            use_gene_attention=use_gene_attention,
            use_multi_head_attention=use_multi_head_attention,
            use_feature_gating=use_feature_gating,
            use_residual_blocks=use_residual_blocks,
            use_layer_norm=use_layer_norm,
            use_gene_relations=use_gene_relations,
        )
        
        # UNet model for flow matching (unchanged)
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
            gene_expr: RNA expression tensor [B, rna_dim]
            gene_mask: Optional gene mask tensor [B, rna_dim]
            
        Returns:
            Predicted velocity field for the flow matching model
        """
        # Encode RNA expression
        rna_embedding = self.rna_encoder(gene_expr, mask=gene_mask)
        
        # Get vector field from UNet model
        return self.unet(x, t, extra={"rna_embedding": rna_embedding})