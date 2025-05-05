import torch
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import needed for the EulerSolver class
from rectified_flow import EulerSolver, DOPRI5Solver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

def train_with_rectified_flow(
    model,
    train_loader,
    val_loader,
    rectified_flow,
    device,
    num_epochs=30,
    lr=1e-4,
    best_model_path="best_rna_to_hne_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
):
    """Train the RNA to H&E cell image generator model with rectified flow"""
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
            gene_mask = batch['gene_mask'].to(device)
            target_images = batch['image'].to(device)

            # Sample random times
            t = torch.rand(gene_expr.shape[0], device=device)
            
            # Get path samples with rectified flow
            path_sample = rectified_flow.sample_path(x_1=target_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]
            
            # Predict vector field with mixed precision
            with torch.amp.autocast('cuda',enabled=use_amp):
                v_pred = model(x_t, t, gene_expr, gene_mask) # <-- Pass both tensors
                l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                loss = rectified_flow.loss_fn(v_pred, target_velocity) + l1_penalty
            
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
                gene_mask = batch['gene_mask'].to(device)
                target_images = batch['image'].to(device)
                
                # Sample random times
                t = torch.rand(gene_expr.shape[0], device=device)
                
                # Get path samples with rectified flow
                path_sample = rectified_flow.sample_path(x_1=target_images, t=t)
                x_t = path_sample["x_t"]
                target_velocity = path_sample["velocity"]
                
                # Predict vector field
                with torch.amp.autocast('cuda', enabled=use_amp):
                    v_pred = model(x_t, t, gene_expr, gene_mask) # <-- Pass both tensors
                    l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                    loss = rectified_flow.loss_fn(v_pred, target_velocity) + l1_penalty
                
                val_loss_metric.update(loss)
        
        val_loss = val_loss_metric.compute().item()
        val_losses.append(val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_config = {
                'rna_dim': model.rna_dim, # Access from model attribute
                'img_channels': model.img_channels, # Access from model attribute
                'img_size': model.img_size, # Access from model attribute
                # Add any other relevant config parameters stored in the model
                # ...
            }

            torch.save({
                'model': model.state_dict(),
                'config': model_config,
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

def generate_images_with_rectified_flow(model, rectified_flow, gene_expr, device, num_steps=100):
    """
    Generate cell images from gene expression profiles using rectified flow and euler solver
    """
    # Create the solver
    solver = DOPRI5Solver(model, rectified_flow)
    
    # Generate images
    generated_images = solver.generate_sample(
        rna_expr=gene_expr,
        num_steps=num_steps,
        device=device
    )
    
    # Denormalize images
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images