import os
import sys
import torch
import logging
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .diffusion import DiffusionSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class NativeScalerWithGradNormCount:
    """Gradient scaling utility for efficient mixed precision training."""
    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')
        self.grad_norm = 0
        
    def __call__(self, loss, optimizer, parameters, update_grad=True):
        # Use the scaler's scale method properly
        scaled_loss = self._scaler.scale(loss)
        scaled_loss.backward()
        
        if update_grad:
            # Unscale gradients before clipping
            self._scaler.unscale_(optimizer)
            self.grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            
        # Step and update
        self._scaler.step(optimizer)
        self._scaler.update()
        optimizer.zero_grad()
            
    def state_dict(self):
        return self._scaler.state_dict()
        
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes for averaging"""
    if world_size == 1:
        return tensor
    
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor = reduced_tensor / world_size
    return reduced_tensor


def train_with_diffusion(
    model,
    train_loader,
    val_loader,
    diffusion,
    device,
    num_epochs=30,
    lr=1e-4,
    best_model_path="best_diffusion_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
    is_multi_cell=False,
    start_epoch=0,
    best_val_loss=float('inf'),
    optimizer=None,
    checkpoint_path=None,
    save_checkpoint_fn=None,
    scaler=None,
    # DDP parameters (backward compatible - defaults maintain single GPU behavior)
    train_sampler=None,
    rank=0,
    world_size=1,
    use_ddp=False
):
    """
    Train the RNA to H&E cell image generator model with diffusion, supporting DDP and checkpoint resume.
    """
    model.to(device)
    
    # If optimizer is not provided, create one
    if optimizer is None:
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
    
    # If scaler is not provided and use_amp is True, create one
    if scaler is None and use_amp:
        scaler = NativeScalerWithGradNormCount()
    
    # Metrics
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    
    train_losses, val_losses = [], []
    
    # Early stopping variables
    counter = 0
    early_stop = False

    # If resuming, step the lr_scheduler to start_epoch
    for _ in range(start_epoch):
        lr_scheduler.step()
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for DistributedSampler
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training
        model.train()
        train_loss_metric.reset()
        
        # Create progress bar only on rank 0
        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        else:
            train_pbar = train_loader

        # Running loss for tqdm display
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_pbar):
            gene_expr = batch['gene_expr'].to(device)
            target_images = batch['image'].to(device)

            # Handle gene mask if present, otherwise set to None
            gene_mask = batch.get('gene_mask', None)
            if gene_mask is not None:
                gene_mask = gene_mask.to(device)

            # Get number of cells if using multi-cell model
            num_cells = None
            if is_multi_cell and 'num_cells' in batch:
                num_cells = batch['num_cells']

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (gene_expr.shape[0],), device=device).long()
            
            # Compute loss with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Use diffusion loss function
                loss = diffusion.loss_fn(
                    model=model,
                    x_0=target_images,
                    t=t,
                    rna_expr=gene_expr,
                    gene_mask=gene_mask,
                    num_cells=num_cells,
                    is_multi_cell=is_multi_cell
                )
                
                # Add L1 regularization for gene weights (if needed)
                if is_multi_cell:
                    # Handle DDP model access
                    if isinstance(model, DDP):
                        l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.cell_encoder[0].weight)) * 0.001
                    else:
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                else:
                    # Handle DDP model access
                    if isinstance(model, DDP):
                        l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.encoder[0].weight)) * 0.001
                    else:
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                
                loss = loss + l1_penalty
            
            # Backpropagation with loss scaling
            if use_amp:
                scaler(
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
            
            # Update running loss for tqdm display (only on rank 0)
            if rank == 0:
                running_loss += loss.item()
                num_batches += 1
                avg_loss = running_loss / num_batches
                train_pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Current': f'{loss.item():.4f}',
                    'LR': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Calculate average training loss
        train_loss = train_loss_metric.compute()
        
        # Reduce loss across all processes if using DDP
        if use_ddp:
            train_loss = reduce_tensor(train_loss, world_size)
        
        train_loss = train_loss.item()
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss_metric.reset()
        
        # Create progress bar only on rank 0
        if rank == 0:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        else:
            val_pbar = val_loader

        # Running validation loss for tqdm display
        running_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                gene_expr = batch['gene_expr'].to(device)
                target_images = batch['image'].to(device)
                
                # Handle gene mask if present, otherwise set to None
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)
                
                # Get number of cells if using multi-cell model
                num_cells = None
                if is_multi_cell and 'num_cells' in batch:
                    num_cells = batch['num_cells']
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.timesteps, (gene_expr.shape[0],), device=device).long()
                
                # Compute validation loss with mixed precision
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Use diffusion loss function
                    loss = diffusion.loss_fn(
                        model=model,
                        x_0=target_images,
                        t=t,
                        rna_expr=gene_expr,
                        gene_mask=gene_mask,
                        num_cells=num_cells,
                        is_multi_cell=is_multi_cell
                    )
                    
                    # Add L1 regularization for gene weights (if needed)
                    if is_multi_cell:
                        # Handle DDP model access
                        if isinstance(model, DDP):
                            l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.cell_encoder[0].weight)) * 0.001
                        else:
                            l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                    else:
                        # Handle DDP model access
                        if isinstance(model, DDP):
                            l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.encoder[0].weight)) * 0.001
                        else:
                            l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                    
                    loss = loss + l1_penalty
                
                val_loss_metric.update(loss)
                
                # Update running validation loss for tqdm display (only on rank 0)
                if rank == 0:
                    running_val_loss += loss.item()
                    num_val_batches += 1
                    avg_val_loss = running_val_loss / num_val_batches
                    val_pbar.set_postfix({
                        'Val Loss': f'{avg_val_loss:.4f}',
                        'Current': f'{loss.item():.4f}'
                    })
        
        # Calculate average validation loss
        val_loss = val_loss_metric.compute()
        
        # Reduce loss across all processes if using DDP
        if use_ddp:
            val_loss = reduce_tensor(val_loss, world_size)
        
        val_loss = val_loss.item()
        val_losses.append(val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Log only on rank 0
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")

        # Synchronize before checkpoint operations
        if use_ddp:
            dist.barrier()

        # Save checkpoint
        if checkpoint_path and save_checkpoint_fn:
            if rank == 0:
                # Extract model state dict properly for DDP
                if isinstance(model, DDP):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                if scaler is not None:
                    checkpoint_state['scaler_state_dict'] = scaler.state_dict()
                # Handle both old and new save_checkpoint_fn signatures
                try:
                    save_checkpoint_fn(checkpoint_state, checkpoint_path, rank)
                except TypeError:
                    # Fallback for old signature without rank parameter
                    save_checkpoint_fn(checkpoint_state, checkpoint_path)
        
        # Save best model and check for early stopping (only on rank 0)
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Extract model state dict properly for DDP
                if isinstance(model, DDP):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()

                model_config = {
                    'rna_dim': model.module.rna_dim if isinstance(model, DDP) else model.rna_dim,
                    'img_channels': model.module.img_channels if isinstance(model, DDP) else model.img_channels,
                    'img_size': model.module.img_size if isinstance(model, DDP) else model.img_size,
                }

                torch.save({
                    'model': model_state_dict,
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

        # Broadcast early stopping decision to all ranks
        if use_ddp:
            early_stop_tensor = torch.tensor(early_stop, dtype=torch.bool, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = early_stop_tensor.item()
        
        if early_stop:
            break
    
    return train_losses, val_losses


def generate_images_with_diffusion(
    model,
    diffusion, 
    gene_expr, 
    device, 
    num_steps=100,
    gene_mask=None,
    num_cells=None,
    is_multi_cell=False,
    method="ddim"  # "ddpm" for standard sampling or "ddim" for accelerated sampling
):
    """
    Generate cell images from gene expression profiles using diffusion model
    
    Args:
        model: The RNA to H&E model (can be DDP wrapped)
        diffusion: The diffusion module
        gene_expr: RNA expression tensor
        device: Computation device
        num_steps: Number of steps for the generation process
        gene_mask: Optional gene mask tensor
        num_cells: Optional number of cells per patch for multi-cell model
        is_multi_cell: Whether using multi-cell model
        method: Sampling method (ddpm or ddim)
        
    Returns:
        Generated images tensor
    """
    # Handle DDP model wrapping for inference
    actual_model = model.module if isinstance(model, DDP) else model
    
    # Initialize the diffusion sampler with the actual model
    sampler = DiffusionSampler(actual_model, diffusion)
    
    # Generate images
    generated_images = sampler.generate_sample(
        rna_expr=gene_expr,
        num_steps=num_steps,
        device=device,
        method=method,
        gene_mask=gene_mask,
        num_cells=num_cells,
        is_multi_cell=is_multi_cell
    )
    
    return generated_images