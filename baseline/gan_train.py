import torch
from torch.optim import Adam
from torch.nn import functional as F
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def train_with_gan(
    gan,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    lr_g=1e-4,
    lr_d=4e-4,
    best_model_path="best_rna_to_hne_gan_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
    d_steps=1,
    g_steps=1,
    is_multi_cell=False,
):
    """Train the RNA to H&E cell image generator model with GAN approach"""
    generator = gan.generator
    discriminator = gan.discriminator
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    # Optimizers with weight decay
    optimizer_G = Adam(
        generator.parameters(),
        lr=lr_g,
        betas=(0.5, 0.999),
        weight_decay=weight_decay
    )
    
    optimizer_D = Adam(
        discriminator.parameters(),
        lr=lr_d,
        betas=(0.5, 0.999),
        weight_decay=weight_decay
    )
    
    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_G,
        T_max=num_epochs,
        eta_min=lr_g * 0.01
    )
    
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_D,
        T_max=num_epochs,
        eta_min=lr_d * 0.01
    )
    
    # Loss scaler for mixed precision
    loss_scaler_G = NativeScalerWithGradNormCount() if use_amp else None
    loss_scaler_D = NativeScalerWithGradNormCount() if use_amp else None
    
    # Metrics
    gen_loss_metric = MeanMetric().to(device)
    disc_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    
    best_val_loss = float('inf')
    train_gen_losses, train_disc_losses, val_losses = [], [], []
    
    # Early stopping variables
    counter = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        # Training
        generator.train()
        discriminator.train()
        gen_loss_metric.reset()
        disc_loss_metric.reset()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # Extract data
            if is_multi_cell:
                # Process multi-cell data
                gene_expr = batch['gene_expr'].to(device)
                real_images = batch['image'].to(device)
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)
                num_cells = batch.get('num_cells', None)
            else:
                # Process single-cell data
                gene_expr = batch['gene_expr'].to(device)
                real_images = batch['image'].to(device)
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)
                num_cells = None
            
            batch_size = gene_expr.shape[0]
            
            # ---------------------------
            # Train Discriminator
            # ---------------------------
            for _ in range(d_steps):
                optimizer_D.zero_grad()
                
                # Process real images
                with torch.amp.autocast('cuda', enabled=use_amp):
                    real_pred = discriminator(real_images, gene_expr)
                    
                    # Generate fake images
                    noise = gan.sample_noise(batch_size, device)
                    if is_multi_cell:
                        fake_images = generator(noise, gene_expr, num_cells, gene_mask)
                    else:
                        fake_images = generator(noise, gene_expr, gene_mask)
                    
                    # Discriminate fake images
                    fake_pred = discriminator(fake_images.detach(), gene_expr)
                    
                    # Calculate discriminator loss
                    d_loss_dict = gan.discriminator_loss(
                        real_pred, fake_pred, real_images, fake_images, gene_expr
                    )
                    d_loss = d_loss_dict["disc_total"]
                
                # Backpropagation with loss scaling
                if use_amp:
                    loss_scaler_D(
                        d_loss,
                        optimizer_D,
                        parameters=discriminator.parameters(),
                        update_grad=True
                    )
                else:
                    d_loss.backward()
                    optimizer_D.step()
                
                disc_loss_metric.update(d_loss)
            
            # ---------------------------
            # Train Generator
            # ---------------------------
            for _ in range(g_steps):
                optimizer_G.zero_grad()
                
                # Generate fake images
                with torch.amp.autocast('cuda', enabled=use_amp):
                    noise = gan.sample_noise(batch_size, device)
                    if is_multi_cell:
                        fake_images = generator(noise, gene_expr, num_cells, gene_mask)
                    else:
                        fake_images = generator(noise, gene_expr, gene_mask)
                    
                    # Discriminate fake images
                    fake_pred = discriminator(fake_images, gene_expr)
                    
                    # Calculate generator loss
                    g_loss_dict = gan.generator_loss(fake_pred, fake_images, real_images)
                    g_loss = g_loss_dict["gen_total"]
                
                # Backpropagation with loss scaling
                if use_amp:
                    loss_scaler_G(
                        g_loss,
                        optimizer_G,
                        parameters=generator.parameters(),
                        update_grad=True
                    )
                else:
                    g_loss.backward()
                    optimizer_G.step()
                
                gen_loss_metric.update(g_loss)
                
            # Print progress periodically
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx} - Gen Loss: {g_loss.item():.4f}, Disc Loss: {d_loss.item():.4f}")
        
        gen_loss = gen_loss_metric.compute().item()
        disc_loss = disc_loss_metric.compute().item()
        
        train_gen_losses.append(gen_loss)
        train_disc_losses.append(disc_loss)
        
        # Validation
        generator.eval()
        discriminator.eval()
        val_loss_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Extract data
                if is_multi_cell:
                    # Process multi-cell data
                    gene_expr = batch['gene_expr'].to(device)
                    real_images = batch['image'].to(device)
                    gene_mask = batch.get('gene_mask', None)
                    if gene_mask is not None:
                        gene_mask = gene_mask.to(device)
                    num_cells = batch.get('num_cells', None)
                else:
                    # Process single-cell data
                    gene_expr = batch['gene_expr'].to(device)
                    real_images = batch['image'].to(device)
                    gene_mask = batch.get('gene_mask', None)
                    if gene_mask is not None:
                        gene_mask = gene_mask.to(device)
                    num_cells = None
                
                batch_size = gene_expr.shape[0]
                
                # Generate fake images
                noise = gan.sample_noise(batch_size, device)
                if is_multi_cell:
                    fake_images = generator(noise, gene_expr, num_cells, gene_mask)
                else:
                    fake_images = generator(noise, gene_expr, gene_mask)
                
                # Evaluate using L1 loss for validation (simple metric)
                val_loss = F.l1_loss(fake_images, real_images)
                val_loss_metric.update(val_loss)
        
        val_loss = val_loss_metric.compute().item()
        val_losses.append(val_loss)
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_config = {
                'rna_dim': generator.base_model.rna_dim,
                'img_channels': generator.img_channels,
                'img_size': generator.img_size,
            }

            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'config': model_config,
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D': lr_scheduler_D.state_dict(),
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
    
    return train_gen_losses, train_disc_losses, val_losses

def generate_images_with_gan(
    gan,
    gene_expr, 
    device, 
    gene_mask=None,
    num_cells=None,
    noise=None,
    num_samples=1
):
    """
    Generate cell images from gene expression profiles using GAN
    
    Args:
        gan: The CellGAN object
        gene_expr: RNA expression tensor
        device: Computation device
        gene_mask: Optional gene mask tensor
        num_cells: Optional number of cells per patch for multi-cell model
        noise: Optional noise tensor
        num_samples: Number of samples to generate per RNA profile
        
    Returns:
        Generated images tensor
    """
    # Ensure models are in eval mode
    gan.generator.eval()
    
    # Move to the correct device
    gene_expr = gene_expr.to(device)
    if gene_mask is not None:
        gene_mask = gene_mask.to(device)
    
    # For generating multiple samples per RNA profile
    if num_samples > 1:
        # Repeat each RNA profile num_samples times
        batch_size = gene_expr.shape[0]
        
        # Create repeated indices
        repeat_indices = torch.arange(batch_size, device=device).repeat_interleave(num_samples)
        
        # Repeat RNA profiles
        gene_expr_repeated = gene_expr[repeat_indices]
        
        # Repeat gene_mask if provided
        if gene_mask is not None:
            gene_mask_repeated = gene_mask[repeat_indices]
        else:
            gene_mask_repeated = None
            
        # Repeat num_cells if provided
        if num_cells is not None:
            num_cells_repeated = [num_cells[i] for i in repeat_indices.cpu().numpy()]
        else:
            num_cells_repeated = None
            
        # Generate the images with different noise for each sample
        with torch.no_grad():
            generated_images = gan.generate_images(
                rna_expr=gene_expr_repeated,
                gene_mask=gene_mask_repeated,
                num_cells=num_cells_repeated
            )
            
        # Reshape to [batch_size, num_samples, channels, height, width]
        generated_images = generated_images.view(
            batch_size, num_samples, 
            generated_images.shape[1], 
            generated_images.shape[2], 
            generated_images.shape[3]
        )
    else:
        # Generate a single sample per RNA profile
        with torch.no_grad():
            generated_images = gan.generate_images(
                rna_expr=gene_expr,
                gene_mask=gene_mask,
                num_cells=num_cells,
                z=noise
            )
    
    return generated_images