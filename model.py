import os
import sys
import torch
import torch.nn as nn
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import RNAConditionedUNet
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

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
        return self.encoder(x)


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

# ======================================
# Training Implementation
# ======================================

class NativeScalerWithGradNormCount:
    """Gradient scaling utility for efficient mixed precision training."""
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
        self.loss_scale = 2**16
        self.inv_scale = 1. / self.loss_scale
        self.grad_norm = 0
        
    def __call__(self, loss, optimizer, parameters, update_grad=True):
        loss = loss * self.loss_scale
        loss.backward()
        
        if update_grad:
            self.grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0) * self.inv_scale
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            
    def state_dict(self):
        return self._scaler.state_dict()
        
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    lr=1e-4,
    best_model_path="best_rna_to_hne_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
):
    """Train the RNA to H&E cell image generator model with advanced techniques"""
    model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01
    )
    
    # Flow matching path
    path = CondOTProbPath()
    
    # Loss scaler for mixed precision
    loss_scaler = NativeScalerWithGradNormCount() if use_amp else None
    
    # Metrics
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    # Early stopping variables
    counter = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_metric.reset()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            gene_expr = batch['gene_expr'].to(device)
            target_images = batch['image'].to(device)
            
            # Sample random times
            t = torch.rand(gene_expr.shape[0], device=device)
            
            # Get path samples
            path_sample = path.sample(t=t, x_0=torch.randn_like(target_images), x_1=target_images)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t
            
            # Predict vector field with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                v_pred = model(x_t, t, gene_expr)
                loss = torch.pow(v_pred - u_t, 2).mean()
            
            # Backpropagation with loss scaling
            if use_amp:
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True
                )
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss_metric.update(loss)
        
        train_loss = train_loss_metric.compute().item()
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                gene_expr = batch['gene_expr'].to(device)
                target_images = batch['image'].to(device)
                
                # Sample random times
                t = torch.rand(gene_expr.shape[0], device=device)
                
                # Get path samples
                path_sample = path.sample(t=t, x_0=torch.randn_like(target_images), x_1=target_images)
                x_t = path_sample.x_t
                u_t = path_sample.dx_t
                
                # Predict vector field
                with torch.cuda.amp.autocast(enabled=use_amp):
                    v_pred = model(x_t, t, gene_expr)
                    loss = torch.pow(v_pred - u_t, 2).mean()
                
                val_loss_metric.update(loss)
        
        val_loss = val_loss_metric.compute().item()
        val_losses.append(val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, best_model_path)
            logger.info(f"Model saved with validation loss: {val_loss:.4f}")
            counter = 0  # Reset counter
        else:
            counter += 1
            logger.info(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break
        
        if early_stop:
            break
    
    return train_losses, val_losses

def generate_images(model, gene_expr, device, num_steps=50, cfg_scale=0.0):
    """
    Generate cell images from gene expression profiles using the trained model
    with advanced ODE solver
    """
    model.eval()
    gene_expr = gene_expr.to(device)
    batch_size = gene_expr.shape[0]
    
    # Create a model wrapper for the ODE solver
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.nfe_counter = 0
            
        def forward(self, x, t):
            # Prepare RNA embedding for each sample in the batch
            with torch.no_grad():
                rna_embedding = self.model.rna_encoder(gene_expr)
            
            # Expand t for the batch
            t_batch = torch.ones(batch_size, device=device) * t
            
            # Get vector field prediction
            with torch.amp.autocast('cuda'):
                v_pred = self.model(x, t_batch, gene_expr)
            
            self.nfe_counter += 1
            return v_pred

        def reset_nfe_counter(self):
            self.nfe_counter = 0
            
        def get_nfe(self):
            return self.nfe_counter
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    # Create ODE solver
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Initialize with random noise
    x_0 = torch.randn(batch_size, model.img_channels, model.img_size, model.img_size, device=device)
    
    # Setup time grid
    time_grid = torch.tensor([0.0, 1.0], device=device)
    
    # Generate images
    with torch.no_grad():
        wrapped_model.reset_nfe_counter()
        x_1 = solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method="euler",
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=0.01,
        )
        
        # Log number of function evaluations
        logger.info(f"Generated images with {wrapped_model.get_nfe()} function evaluations")
    
    # # Denormalize images
    # x_1 = x_1 * 0.5 + 0.5
    x_1 = torch.clamp(x_1, 0, 1)
    
    return x_1