import os
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import RNAConditionedUNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# RNA Encoder
# ======================================

class RNAEncoder(nn.Module):
    """
    Encoder for RNA expression data that produces embeddings for conditioning the UNet.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128):
        super().__init__()
        # Add attention weights for genes
        self.gene_attention = nn.Parameter(torch.ones(input_dim) / input_dim)
        # Rest of encoder remains the same
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply attention to genes
        attention = F.softmax(self.gene_attention, dim=0)
        x_weighted = x * attention
        return self.encoder(x_weighted)
        
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
    ):
        super().__init__()
        
        # RNA expression encoder
        self.rna_encoder = RNAEncoder(
            input_dim=rna_dim,
            hidden_dims=[512, 256],
            output_dim=model_channels * 4  # Match time_embed_dim
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
        
        self.img_channels = img_channels
        self.img_size = img_size
        
    def forward(self, x, t, rna_expr):
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
        rna_embedding = self.rna_encoder(rna_expr)
        
        # Get vector field from UNet model
        return self.unet(x, t, extra={"rna_embedding": rna_embedding})