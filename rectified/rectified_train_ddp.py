import os
import sys
import torch
import logging
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rectified.rectified_flow import DOPRI5Solver
from src.utils import manage_checkpoints
from src.spatial_graph_loss import SpatialGraphLossModule
from rectified.utils import generate_images_with_rectified_flow

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


def train_with_rectified_flow(
    model,
    train_loader,
    val_loader,
    rectified_flow,
    device,
    num_epochs=30,
    lr=1e-4,
    best_model_path="best_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
    is_multi_cell=False,
    start_epoch=0,
    best_val_loss=float('inf'),
    optimizer=None,
    lr_scheduler=None,
    checkpoint_path=None,
    save_checkpoint_fn=None,
    scaler=None,
    train_sampler=None,
    rank=0,
    world_size=1,
    use_ddp=False,
    checkpoint_freq=5,
    no_wandb=False,
    log_interval_pct=1.0,
    max_checkpoints=5,
    use_spatial_loss=False,
    spatial_loss_method='simple',
    spatial_loss_weight=0.1,
    spatial_loss_k_neighbors=5,
    spatial_loss_start_epoch=None,
    spatial_loss_start_val_loss=None,
    spatial_loss_warmup_epochs=5,
    spatial_patience=None,
    force_spatial_loss_from_resume=False,
    resumed_checkpoint_state=None,
):
    """
    Train the RNA to H&E cell image generator model with rectified flow, supporting DDP and checkpoint resume.
    
    New Args:
        use_spatial_loss: Whether to use spatial graph loss
        spatial_loss_weight: Weight for spatial graph loss term
        spatial_loss_k_neighbors: Number of neighbors for spatial graph construction
        spatial_loss_start_epoch: Epoch to start spatial loss (overrides start_val_loss)
        spatial_loss_start_val_loss: Start spatial loss when validation loss drops below this
        spatial_loss_warmup_epochs: Number of epochs to warmup spatial loss weight
        force_spatial_loss_from_resume: If True and resuming, start spatial loss immediately
        resumed_checkpoint_state: Dict containing checkpoint info (for spatial loss state)
    """
    from src.spatial_graph_loss import SpatialGraphLossModule
    
    model.to(device)

    # If optimizer is not provided, create one
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )

    # Create learning rate scheduler only if not provided
    if lr_scheduler is None:
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
    current_patience = patience
    
    spatial_loss_was_active = False

    if spatial_patience is None:
        spatial_patience = patience

    # Initialize spatial loss module if enabled
    spatial_loss_module = None
    spatial_loss_enabled = False
    spatial_loss_actual_start_epoch = None
    
    # Check if we're resuming with spatial loss info
    resumed_spatial_info = None
    if resumed_checkpoint_state is not None:
        resumed_spatial_info = {
            'was_enabled': resumed_checkpoint_state.get('spatial_loss_enabled', False),
            'previous_start_epoch': resumed_checkpoint_state.get('spatial_loss_start_epoch', None)
        }
        
        if rank == 0 and resumed_spatial_info['was_enabled']:
            logger.info(f"Resumed checkpoint had spatial loss enabled from epoch {resumed_spatial_info['previous_start_epoch']}")
    
    if use_spatial_loss and is_multi_cell:
        try:
            spatial_loss_module = SpatialGraphLossModule(
                method=spatial_loss_method,  # 'simple' or 'segmentation'
                device=device, 
                k_neighbors=spatial_loss_k_neighbors,
                warmup_epochs=spatial_loss_warmup_epochs,
                start_epoch=spatial_loss_actual_start_epoch if spatial_loss_actual_start_epoch is not None else 999999,
                gradient_weight=1.0,  # For simple method
                texture_weight=0.5,   # For simple method
            )
            
            if rank == 0:
                if spatial_loss_actual_start_epoch is not None:
                    logger.info(f"Spatial graph loss ({spatial_loss_method}) configured to start at epoch {spatial_loss_actual_start_epoch} "
                            f"with {spatial_loss_warmup_epochs} epochs warmup")
                else:
                    logger.info(f"Spatial graph loss ({spatial_loss_method}) will start when val_loss < {spatial_loss_start_val_loss}")
                    
            # Determine when to start spatial loss
            if force_spatial_loss_from_resume and start_epoch > 0:
                # Force spatial loss to start immediately when resuming
                spatial_loss_actual_start_epoch = start_epoch
                spatial_loss_enabled = True  # Mark as enabled immediately
                spatial_loss_was_active = True  # Mark as active to prevent double-reset
                # Reset best_val_loss and counter immediately
                best_val_loss = float('inf')
                counter = 0
                current_patience = spatial_patience
                if rank == 0:
                    logger.info("=" * 80)
                    logger.info(f"🔄 SPATIAL LOSS FORCE-ACTIVATED FROM RESUME")
                    logger.info(f"   - Starting from epoch {start_epoch}")
                    logger.info(f"   - Warmup epochs: {spatial_loss_warmup_epochs}")
                    logger.info(f"   - Resetting best validation loss tracking")
                    logger.info(f"   - New patience: {spatial_patience} epochs")
                    logger.info("=" * 80)
            elif resumed_spatial_info and resumed_spatial_info['was_enabled']:
                # Continue with spatial loss if it was already enabled
                spatial_loss_actual_start_epoch = resumed_spatial_info['previous_start_epoch']
                spatial_loss_enabled = True
                if rank == 0:
                    logger.info(f"Continuing spatial loss from previous training (started at epoch {spatial_loss_actual_start_epoch})")
            elif spatial_loss_start_epoch is not None:
                spatial_loss_actual_start_epoch = spatial_loss_start_epoch
            elif spatial_loss_start_val_loss is not None:
                # Will be determined dynamically based on val loss
                spatial_loss_actual_start_epoch = None
            else:
                # Default: start after 70% of epochs
                spatial_loss_actual_start_epoch = int(num_epochs * 0.7)
            
            if rank == 0:
                if spatial_loss_actual_start_epoch is not None:
                    logger.info(f"Spatial graph loss configured to start at epoch {spatial_loss_actual_start_epoch} "
                              f"with {spatial_loss_warmup_epochs} epochs warmup")
                else:
                    logger.info(f"Spatial graph loss will start when val_loss < {spatial_loss_start_val_loss}")
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to initialize spatial loss: {e}")
                import traceback
                traceback.print_exc()
            spatial_loss_module = None

    # If resuming, step the lr_scheduler to start_epoch
    if start_epoch > 0:
        # Get the original T_max from checkpoint if available
        original_T_max = resumed_checkpoint_state.get('lr_scheduler_T_max', start_epoch) if resumed_checkpoint_state is not None else start_epoch
        
        if num_epochs > original_T_max:
            # Extending training beyond original plan
            remaining_epochs = num_epochs - start_epoch
            if rank == 0:
                logger.info(f"Extending training: originally {original_T_max} epochs, "
                           f"now training until epoch {num_epochs}")
                logger.info(f"Creating new LR schedule for remaining {remaining_epochs} epochs")
            
            # Create a fresh scheduler for the remaining epochs
            # This gives a smooth continuation with a new cosine curve
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=remaining_epochs,
                eta_min=lr * 0.01
            )
        else:
            # Normal resumption within original epoch plan
            # Scheduler state was already loaded, just verify
            if rank == 0:
                logger.info(f"Resuming with original LR schedule (T_max={num_epochs})")

    for epoch in range(start_epoch, num_epochs):
        # Check if spatial loss should be enabled based on validation loss threshold
        if (spatial_loss_module is not None and 
            not spatial_loss_enabled and 
            spatial_loss_start_val_loss is not None and 
            len(val_losses) > 0 and 
            val_losses[-1] < spatial_loss_start_val_loss):
            
            spatial_loss_enabled = True
            spatial_loss_module.start_epoch = epoch
            if rank == 0:
                logger.info(f"Spatial graph loss enabled at epoch {epoch} "
                          f"(val_loss {val_losses[-1]:.4f} < threshold {spatial_loss_start_val_loss})")
        
        # Set epoch for DistributedSampler
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training
        model.train()
        train_loss_metric.reset()

        # Create progress bar only on rank 0
        if rank == 0:
            update_interval = max(1, int(len(train_loader) * log_interval_pct / 100))
            train_pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs} - Training",
                disable=False
            )

        # Running loss for tqdm display
        running_loss = 0.0
        running_base_loss = 0.0
        running_spatial_loss = 0.0
        spatial_loss_weight_current = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            gene_expr = batch['gene_expr'].to(device)
            target_images = batch['image'].to(device)

            gene_mask = batch.get('gene_mask', None)
            if gene_mask is not None:
                gene_mask = gene_mask.to(device)

            num_cells = None
            if is_multi_cell and 'num_cells' in batch:
                num_cells = batch['num_cells']

            t = torch.rand(gene_expr.shape[0], device=device)
            path_sample = rectified_flow.sample_path(x_1=target_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]

            with torch.amp.autocast('cuda', enabled=use_amp):
                if is_multi_cell:
                    v_pred = model(x_t, t, gene_expr, num_cells, gene_mask)
                    if isinstance(model, DDP):
                        l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.cell_encoder[0].weight)) * 0.001
                    else:
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                else:
                    v_pred = model(x_t, t, gene_expr, gene_mask)
                    if isinstance(model, DDP):
                        l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.encoder[0].weight)) * 0.001
                    else:
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001

                base_loss = rectified_flow.loss_fn(v_pred, target_velocity) + l1_penalty
                
                # Add spatial graph loss if enabled
                spatial_loss_value = 0.0
                if spatial_loss_module is not None and is_multi_cell:
                    try:
                        # Approximate generated image from velocity
                        approx_generated = x_t + v_pred * (1.0 - t.view(-1, 1, 1, 1))
                        approx_generated = torch.clamp(approx_generated, 0, 1)
                        
                        approx_generated_f32 = approx_generated.detach().float()
                        target_images_f32 = target_images.detach().float()
                        
                        with torch.cuda.amp.autocast(enabled=False):
                            spatial_loss_tensor, weight = spatial_loss_module(
                                approx_generated_f32, target_images_f32, epoch
                            )
                        spatial_loss_value = spatial_loss_tensor.item()
                        spatial_loss_weight_current = weight
                        
                        if weight > 0:
                            loss = base_loss + (spatial_loss_weight * weight) * spatial_loss_tensor
                        else:
                            loss = base_loss
                    except Exception as e:
                        if rank == 0 and batch_idx % 100 == 0:
                            logger.warning(f"Spatial loss computation failed: {e}")
                        loss = base_loss
                        spatial_loss_value = 0.0
                else:
                    loss = base_loss

            if use_amp:
                scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_metric.update(loss)
            
            if rank == 0:
                running_loss += loss.item()
                running_base_loss += base_loss.item()
                running_spatial_loss += spatial_loss_value
                num_batches += 1
                
                if num_batches % update_interval == 0 or num_batches == len(train_loader):
                    if num_batches == len(train_loader):
                        steps_to_update = len(train_loader) - train_pbar.n
                    else:
                        steps_to_update = update_interval
                    
                    train_pbar.update(steps_to_update)
                    
                    avg_loss = running_loss / num_batches
                    avg_base_loss = running_base_loss / num_batches
                    avg_spatial_loss = running_spatial_loss / num_batches
                    
                    postfix_dict = {
                        'Loss': f'{avg_loss:.4f}',
                        'Base': f'{avg_base_loss:.4f}',
                        'LR': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                    }
                    
                    if spatial_loss_weight_current > 0:
                        postfix_dict['Spatial'] = f'{avg_spatial_loss:.4f}'
                        postfix_dict['SW'] = f'{spatial_loss_weight_current:.2f}'
                    
                    train_pbar.set_postfix(postfix_dict)

        if rank == 0:
            train_pbar.close()

        # Calculate average training loss
        train_loss = train_loss_metric.compute()
        if use_ddp:
            train_loss = reduce_tensor(train_loss, world_size)
        train_loss = train_loss.item()
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss_metric.reset()

        # Create progress bar only on rank 0
        if rank == 0:
            # Update every log_interval_pct% of total batches
            val_update_interval = max(1, int(len(val_loader) * log_interval_pct / 100))
            val_pbar = tqdm(
                total=len(val_loader),
                desc=f"Epoch {epoch+1}/{num_epochs} - Validation",
                disable=False
            )

        # Running validation loss for tqdm display
        running_val_loss = 0.0
        running_val_base_loss = 0.0
        running_val_spatial_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                gene_expr = batch['gene_expr'].to(device)
                target_images = batch['image'].to(device)

                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)

                num_cells = None
                if is_multi_cell and 'num_cells' in batch:
                    num_cells = batch['num_cells']

                t = torch.rand(gene_expr.shape[0], device=device)
                path_sample = rectified_flow.sample_path(x_1=target_images, t=t)
                x_t = path_sample["x_t"]
                target_velocity = path_sample["velocity"]

                with torch.amp.autocast('cuda', enabled=use_amp):
                    if is_multi_cell:
                        v_pred = model(x_t, t, gene_expr, num_cells, gene_mask)
                        # Handle DDP model access
                        if isinstance(model, DDP):
                            l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.cell_encoder[0].weight)) * 0.001
                        else:
                            l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                    else:
                        v_pred = model(x_t, t, gene_expr, gene_mask)
                        # Handle DDP model access
                        if isinstance(model, DDP):
                            l1_penalty = torch.sum(torch.abs(model.module.rna_encoder.encoder[0].weight)) * 0.001
                        else:
                            l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001

                    base_loss = rectified_flow.loss_fn(v_pred, target_velocity) + l1_penalty
                    
                    # Add spatial graph loss if enabled
                    spatial_loss_value = 0.0
                    if spatial_loss_module is not None and is_multi_cell:
                        try:
                            # Approximate generated image from velocity
                            approx_generated = x_t + v_pred * (1.0 - t.view(-1, 1, 1, 1))
                            approx_generated = torch.clamp(approx_generated, 0, 1)
                            
                            approx_generated_f32 = approx_generated.detach().float()
                            target_images_f32 = target_images.detach().float()
                            
                            with torch.cuda.amp.autocast(enabled=False):
                                spatial_loss_tensor, weight = spatial_loss_module(
                                    approx_generated_f32, target_images_f32, epoch
                                )
                            
                            spatial_loss_value = spatial_loss_tensor.item()
                            spatial_loss_weight_current = weight
                            
                            if weight > 0:
                                loss = base_loss + (spatial_loss_weight * weight) * spatial_loss_tensor
                            else:
                                loss = base_loss
                        except Exception as e:
                            if rank == 0 and batch_idx % 100 == 0:
                                logger.warning(f"Spatial loss computation failed: {e}")
                                import traceback
                                traceback.print_exc()
                            loss = base_loss
                            spatial_loss_value = 0.0
                    else:
                        loss = base_loss

                val_loss_metric.update(loss)
                
                # Update running validation loss for tqdm display (only on rank 0)
                if rank == 0:
                    running_val_loss += loss.item()
                    running_val_base_loss += base_loss.item()
                    running_val_spatial_loss += spatial_loss_value
                    num_val_batches += 1
                    
                    # Only update display every val_update_interval batches
                    if num_val_batches % val_update_interval == 0 or num_val_batches == len(val_loader):
                        # Update progress bar by val_update_interval steps (or remaining steps)
                        if num_val_batches == len(val_loader):
                            # Last batch - update remaining
                            steps_to_update = len(val_loader) - val_pbar.n
                        else:
                            steps_to_update = val_update_interval
                        
                        val_pbar.update(steps_to_update)
                        
                        avg_val_loss = running_val_loss / num_val_batches
                        avg_val_base_loss = running_val_base_loss / num_val_batches
                        avg_val_spatial_loss = running_val_spatial_loss / num_val_batches
                        
                        postfix_dict = {
                            'Val Loss': f'{avg_val_loss:.4f}',
                            'Base': f'{avg_val_base_loss:.4f}'
                        }
                        
                        if spatial_loss_weight_current > 0:
                            postfix_dict['Spatial'] = f'{avg_val_spatial_loss:.4f}'
                        
                        val_pbar.set_postfix(postfix_dict)
        
        if rank == 0:
            val_pbar.close()

        # Calculate average validation loss
        val_loss = val_loss_metric.compute()
        if use_ddp:
            val_loss = reduce_tensor(val_loss, world_size)
        val_loss = val_loss.item()
        val_losses.append(val_loss)

        if rank == 0:
            postfix_dict = {
                'Val Loss': f'{val_loss:.4f}',
                'Base': f'{running_val_base_loss / max(num_val_batches, 1):.4f}'
            }
            if spatial_loss_weight_current > 0:
                postfix_dict['Spatial'] = f'{running_val_spatial_loss / max(num_val_batches, 1):.4f}'
            val_pbar.set_postfix(postfix_dict)
            val_pbar.close()

        lr_scheduler.step()

        # Log only on rank 0
        if rank == 0:
            # Calculate component averages
            avg_train_base = running_base_loss / max(num_batches, 1)
            avg_val_base = running_val_base_loss / max(num_val_batches, 1)
            
            # Build log message
            log_msg = (f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"LR: {lr_scheduler.get_last_lr()[0]:.6f}")
            
            # Add spatial loss details if module exists (even if weight is 0)
            if spatial_loss_module is not None:
                avg_train_spatial = running_spatial_loss / max(num_batches, 1)
                avg_val_spatial = running_val_spatial_loss / max(num_val_batches, 1)
                
                if spatial_loss_weight_current > 0:
                    # During active spatial loss phase
                    log_msg += (f"\n           Train: Base={avg_train_base:.4f}, "
                               f"Spatial={avg_train_spatial:.4f} (Raw), "
                               f"Weighted={(avg_train_spatial * spatial_loss_weight * spatial_loss_weight_current):.4f}, "
                               f"Weight={spatial_loss_weight_current:.3f}")
                    log_msg += (f"\n           Val:   Base={avg_val_base:.4f}, "
                               f"Spatial={avg_val_spatial:.4f} (Raw), "
                               f"Weighted={(avg_val_spatial * spatial_loss_weight * spatial_loss_weight_current):.4f}")
                else:
                    # Before spatial loss activates
                    log_msg += f"\n           (Spatial loss: inactive, will activate at epoch {spatial_loss_module.start_epoch})"
            
            logger.info(log_msg)
            
            # WandB logging
            if not no_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_base_loss": avg_train_base,
                    "val_base_loss": avg_val_base,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "grad_norm": scaler.grad_norm if scaler is not None else 0,
                }
                
                # Always log spatial components if module exists
                if spatial_loss_module is not None:
                    avg_train_spatial = running_spatial_loss / max(num_batches, 1)
                    avg_val_spatial = running_val_spatial_loss / max(num_val_batches, 1)
                    
                    log_dict["spatial_loss_weight"] = spatial_loss_weight_current
                    log_dict["train_spatial_loss_raw"] = avg_train_spatial
                    log_dict["val_spatial_loss_raw"] = avg_val_spatial
                    log_dict["train_spatial_loss_weighted"] = avg_train_spatial * spatial_loss_weight * spatial_loss_weight_current
                    log_dict["val_spatial_loss_weighted"] = avg_val_spatial * spatial_loss_weight * spatial_loss_weight_current
                    log_dict["spatial_loss_active"] = 1 if spatial_loss_weight_current > 0 else 0
                
                wandb.log(log_dict)
            
        if spatial_loss_weight_current > 0 and not spatial_loss_was_active:
            spatial_loss_was_active = True
            counter = 0
            current_patience = spatial_patience
            
            # Reset best_val_loss to force saving first spatial checkpoint
            best_val_loss_before_spatial = best_val_loss
            best_val_loss = float('inf')
            
            if rank == 0:
                logger.info("=" * 80)
                logger.info(f"🔄 SPATIAL LOSS ACTIVATED - New training phase starting")
                logger.info(f"   - Resetting early stopping counter")
                logger.info(f"   - New patience: {spatial_patience} epochs (was {patience})")
                logger.info(f"   - Resetting best validation loss tracking")
                logger.info(f"   - Previous best val loss: {best_val_loss_before_spatial:.4f}")
                logger.info("=" * 80)
                
                if not no_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "spatial_loss_activated": 1,
                        "early_stopping_reset": 1,
                        "new_patience": spatial_patience,
                        "best_val_loss_before_spatial": best_val_loss_before_spatial
                    })
            
        # Synchronize before checkpoint operations
        if use_ddp:
            dist.barrier()

        # Save checkpoint only on improvement (only on rank 0)
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Extract model state dict properly for DDP
            if isinstance(model, DDP):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            # Create checkpoints subdirectory
            checkpoint_dir = os.path.join(os.path.dirname(best_model_path), "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Determine checkpoint naming based on spatial loss status
            spatial_suffix = "_spatial" if spatial_loss_weight_current > 0 else ""
            checkpoint_name = f"checkpoint_epoch_{epoch+1}_loss_{val_loss:.4f}{spatial_suffix}.pt"
            checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'lr_scheduler_T_max': num_epochs,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'spatial_loss_enabled': (spatial_loss_weight_current > 0),
                'spatial_loss_start_epoch': spatial_loss_actual_start_epoch if spatial_loss_weight_current > 0 else None,
            }
            if scaler is not None:
                checkpoint_state['scaler_state_dict'] = scaler.state_dict()
            
            # Save the checkpoint
            torch.save(checkpoint_state, checkpoint_file)
            logger.info(f"Saved checkpoint: {checkpoint_name} (val_loss: {val_loss:.4f})")
            
            # Update pointers - use spatial suffix for separate tracking
            latest_link = os.path.join(checkpoint_dir, f"latest_checkpoint{spatial_suffix}.pt")
            best_link = os.path.join(checkpoint_dir, f"best_checkpoint{spatial_suffix}.pt")
            
            for link_path in [latest_link, best_link]:
                if os.path.exists(link_path) or os.path.islink(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(checkpoint_name, link_path)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(checkpoint_file, link_path)
            
            logger.info(f"Best model checkpoint saved with validation loss: {val_loss:.4f}")
            
            # Manage checkpoint count separately for spatial and non-spatial
            manage_checkpoints(checkpoint_dir, max_checkpoints, rank, suffix=spatial_suffix)
            
            counter = 0
        elif rank == 0:
            if spatial_loss_module is not None and spatial_loss_was_active:
                epochs_since_spatial_start = epoch - spatial_loss_module.start_epoch
                spatial_in_warmup = (0 <= epochs_since_spatial_start < spatial_loss_module.warmup_epochs)
            else:
                spatial_in_warmup = False

            if spatial_in_warmup:
                logger.info(f"Spatial loss warmup in progress (epoch {epochs_since_spatial_start + 1}/{spatial_loss_module.warmup_epochs}, "
                        f"weight: {spatial_loss_weight_current:.3f}), early stopping paused")
            else:
                counter += 1
                # Use current_patience (which changes when spatial loss activates)
                patience_msg = f"EarlyStopping counter: {counter} out of {current_patience}"
                if spatial_loss_was_active:
                    patience_msg += " (spatial phase)"
                logger.info(patience_msg)
                
                if counter >= current_patience:
                    phase = "spatial training" if spatial_loss_was_active else "training"
                    logger.info(f"Early stopping triggered after {epoch+1} epochs ({phase})")
                    early_stop = True

        # Broadcast early stopping decision to all ranks
        if use_ddp:
            early_stop_tensor = torch.tensor(early_stop, dtype=torch.bool, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = early_stop_tensor.item()

        if early_stop:
            break

    return train_losses, val_losses

