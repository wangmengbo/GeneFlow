import os
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchmetrics.aggregation import MeanMetric
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import from your existing files
from rectified.rectified_train import NativeScalerWithGradNormCount, DOPRI5Solver, generate_images_with_rectified_flow
from waddington.waddington_energy import WaddingtonEnergy
from waddington.waddington_rectified_flow import WaddingtonRectifiedFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def train_with_waddington_guidance(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    lr=1e-4,
    best_model_path="best_waddington_model.pt",
    patience=10,
    use_amp=True,
    weight_decay=0.0,
    is_multi_cell=False,
    # Waddington-specific parameters
    energy_weight=0.1,
    energy_scale=1.0,
    num_attractor_states=5,
    hidden_dim=128,
    # Output directory for visualizations
    output_dir="output_waddington",
):
    """
    Train the RNA to H&E cell image generator model with Waddington landscape energy guidance
    
    This is a drop-in replacement for train_with_rectified_flow that adds biological plausibility
    through Waddington energy guidance.
    """
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the Waddington-guided rectified flow
    waddington_flow = WaddingtonRectifiedFlow(
        sigma_min=0.002,
        sigma_max=80.0,
        energy_weight=energy_weight
    )
    
    # Initialize the Waddington energy model
    gene_dim = model.rna_dim
    waddington_energy = WaddingtonEnergy(
        gene_dim=gene_dim,
        hidden_dim=hidden_dim,
        energy_scale=energy_scale,
        num_attractor_states=num_attractor_states
    ).to(device)
    
    # Set the energy model in the flow
    waddington_flow.waddington_energy = waddington_energy
    
    # Optimizer with weight decay for both model and energy network
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(waddington_energy.parameters()),
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
    train_flow_loss_metric = MeanMetric().to(device)
    train_energy_metric = MeanMetric().to(device)
    
    val_loss_metric = MeanMetric().to(device)
    val_flow_loss_metric = MeanMetric().to(device)
    val_energy_metric = MeanMetric().to(device)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_energy_losses, val_energy_losses = [], []
    
    # Early stopping variables
    counter = 0
    early_stop = False
    
    # Lists to store attractor states for visualization
    attractor_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        waddington_energy.train()
        
        train_loss_metric.reset()
        train_flow_loss_metric.reset()
        train_energy_metric.reset()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
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

            # Sample random times
            t = torch.rand(gene_expr.shape[0], device=device)
            
            # Get path samples with Waddington-guided rectified flow
            path_sample = waddington_flow.sample_path_with_guidance(
                x_1=target_images, 
                t=t,
                gene_expr=gene_expr
            )
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]
            
            # Predict vector field with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Pass num_cells if using multi-cell model
                if is_multi_cell:
                    v_pred = model(x_t, t, gene_expr, num_cells, gene_mask)
                    l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                else:
                    v_pred = model(x_t, t, gene_expr, gene_mask)
                    l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                
                # Calculate loss with Waddington energy guidance
                loss, flow_loss, energy_term = waddington_flow.loss_fn(
                    v_pred, target_velocity, gene_expr, x_t, t
                )
                loss = loss + l1_penalty
            
            # Backpropagation with loss scaling
            if use_amp:
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=list(model.parameters()) + list(waddington_energy.parameters()),
                    update_grad=True
                )
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss_metric.update(loss)
            train_flow_loss_metric.update(flow_loss)
            train_energy_metric.update(energy_term)
        
        train_loss = train_loss_metric.compute().item()
        train_flow_loss = train_flow_loss_metric.compute().item()
        train_energy_loss = train_energy_metric.compute().item()
        
        train_losses.append(train_loss)
        train_energy_losses.append(train_energy_loss)
        
        # Validation
        model.eval()
        waddington_energy.eval()
        
        val_loss_metric.reset()
        val_flow_loss_metric.reset()
        val_energy_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
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
                
                # Sample random times
                t = torch.rand(gene_expr.shape[0], device=device)
                
                # Get path samples with Waddington-guided rectified flow
                path_sample = waddington_flow.sample_path_with_guidance(
                    x_1=target_images, 
                    t=t,
                    gene_expr=gene_expr
                )
                x_t = path_sample["x_t"]
                target_velocity = path_sample["velocity"]
                
                # Predict vector field
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Use same approach as training - handle multi-cell case
                    if is_multi_cell:
                        v_pred = model(x_t, t, gene_expr, num_cells, gene_mask)
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.cell_encoder[0].weight)) * 0.001
                    else:
                        v_pred = model(x_t, t, gene_expr, gene_mask)
                        l1_penalty = torch.sum(torch.abs(model.rna_encoder.encoder[0].weight)) * 0.001
                    
                    # Calculate loss with Waddington energy
                    loss, flow_loss, energy_term = waddington_flow.loss_fn(
                        v_pred, target_velocity, gene_expr, x_t, t
                    )
                    loss = loss + l1_penalty
                
                val_loss_metric.update(loss)
                val_flow_loss_metric.update(flow_loss)
                val_energy_metric.update(energy_term)
        
        val_loss = val_loss_metric.compute().item()
        val_flow_loss = val_flow_loss_metric.compute().item()
        val_energy_loss = val_energy_metric.compute().item()
        
        val_losses.append(val_loss)
        val_energy_losses.append(val_energy_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Log training information
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f} (Flow: {train_flow_loss:.4f}, Energy: {train_energy_loss:.4f})")
        logger.info(f"  Val Loss: {val_loss:.4f} (Flow: {val_flow_loss:.4f}, Energy: {val_energy_loss:.4f})")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model with additional Waddington-specific components
            torch.save({
                'model': model.state_dict(),
                'waddington_energy': waddington_energy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'energy_weight': energy_weight,
                'energy_scale': energy_scale,
                'num_attractor_states': num_attractor_states,
                'model_type': 'multi' if is_multi_cell else 'single',
                'rna_dim': model.rna_dim,
                'img_channels': model.img_channels,
                'img_size': model.img_size,
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
        
        # Store attractor states for visualization
        attractor_states = waddington_energy.get_attractor_states()
        attractor_history.append(attractor_states)
        
        # Periodically visualize the Waddington landscape
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_waddington_landscape(
                waddington_energy, 
                attractor_history, 
                train_losses, val_losses,
                train_energy_losses, val_energy_losses,
                epoch, 
                output_dir
            )
        
        if early_stop:
            break
    
    # Final visualization
    visualize_waddington_landscape(
        waddington_energy, 
        attractor_history, 
        train_losses, val_losses,
        train_energy_losses, val_energy_losses,
        epoch, 
        output_dir,
        is_final=True
    )
    
    # Run landscape analysis on validation data
    if len(val_loader) > 0:
        logger.info("Analyzing Waddington landscape on validation data...")
        analyze_validation_landscape(
            waddington_energy,
            val_loader,
            device,
            output_dir,
            is_multi_cell=is_multi_cell
        )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_energy_losses': train_energy_losses,
        'val_energy_losses': val_energy_losses,
        'attractor_history': attractor_history,
        'waddington_energy': waddington_energy
    }

def generate_images_with_waddington(
    model,
    waddington_flow,
    gene_expr,
    device,
    num_steps=100,
    gene_mask=None,
    num_cells=None,
    is_multi_cell=False,
    guidance_strength=1.0
):
    """
    Generate cell images from gene expression with Waddington landscape guidance
    
    This extends generate_images_with_rectified_flow to use Waddington energy guidance
    """
    # Create wrapper for Waddington-guided generation
    class WaddingtonGuidedModelWrapper:
        def __init__(self, model, guidance_strength=1.0):
            self.model = model
            self.img_channels = model.img_channels
            self.img_size = model.img_size
            self.guidance_strength = guidance_strength
            
        def __call__(self, x, t, rna_expr):
            # Get base velocity from the model
            if is_multi_cell:
                v_base = self.model(x, t, rna_expr, num_cells, gene_mask)
            else:
                v_base = self.model(x, t, rna_expr, gene_mask)
            
            # Apply Waddington energy guidance if available
            if waddington_flow.waddington_energy is not None:
                try:
                    # Calculate energy gradient for guidance
                    x_requires_grad = x.detach().clone().requires_grad_(True)
                    
                    # The energy function will handle channel mismatches internally
                    energy = waddington_flow.waddington_energy(rna_expr, x_requires_grad, t)
                    
                    if energy.sum().requires_grad:
                        energy_sum = torch.sum(energy)
                        
                        # Calculate gradient of energy with respect to x
                        grad_energy = torch.autograd.grad(
                            energy_sum, 
                            x_requires_grad, 
                            create_graph=False, 
                            allow_unused=True,
                            retain_graph=False
                        )[0]
                        
                        if grad_energy is not None:
                            # Scale guidance based on time (stronger early, weaker later)
                            t_expanded = t.view(-1, *([1] * (len(x.shape) - 1)))
                            time_factor = (1 - t_expanded) * self.guidance_strength
                            
                            # Modify velocity to follow energy gradient
                            v_guided = v_base - time_factor * grad_energy
                            return v_guided
                except Exception as e:
                    print(f"Error in energy guidance: {str(e)}, using original velocity")
            
            return v_base
    
    # Create model wrapper with guidance
    model_wrapper = WaddingtonGuidedModelWrapper(model, guidance_strength=guidance_strength)
    
    # Create solver with the guided model
    solver = DOPRI5Solver(model_wrapper, waddington_flow)
    
    # Generate images
    generated_images = solver.generate_sample(
        rna_expr=gene_expr,
        num_steps=num_steps,
        device=device
    )
    
    # Normalize images
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images

def visualize_waddington_landscape(
    waddington_energy, 
    attractor_history, 
    train_losses, val_losses,
    train_energy_losses, val_energy_losses,
    epoch, 
    output_dir,
    is_final=False
):
    """
    Visualize the Waddington landscape and training progress
    """
    # Create directory for visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_energy_losses, label='Train Energy')
    plt.plot(val_energy_losses, label='Val Energy')
    plt.xlabel('Epoch')
    plt.ylabel('Energy Value')
    plt.legend()
    plt.title('Waddington Energy Term')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_epoch_{epoch}.png"))
    if is_final:
        plt.savefig(os.path.join(output_dir, "final_training_curves.png"))
    plt.close()
    
    # Plot attractor states evolution using PCA
    if len(attractor_history) > 0:
        # Get the latest attractor states
        latest_attractors = attractor_history[-1]
        num_attractors = latest_attractors.shape[0]
        gene_dim = latest_attractors.shape[1]
        
        # Skip visualization if gene dimension is too small
        if gene_dim <= 2:
            return
        
        # Combine all attractor states for PCA
        all_attractors = torch.cat(attractor_history, dim=0).numpy()
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        all_attractors_2d = pca.fit_transform(all_attractors)
        
        # Split back by epoch
        attractors_2d_by_epoch = np.split(all_attractors_2d, len(attractor_history))
        
        # Plot attractor evolution
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, num_attractors))
        
        for i in range(num_attractors):
            # Extract trajectory for this attractor
            trajectory = np.array([attractors_2d_by_epoch[e][i] for e in range(len(attractor_history))])
            
            # Plot trajectory with color gradient to show evolution
            for j in range(1, len(trajectory)):
                alpha = 0.5 * (j / len(trajectory)) + 0.5  # Increasing opacity
                plt.plot(
                    trajectory[j-1:j+1, 0], 
                    trajectory[j-1:j+1, 1], 
                    color=colors[i], 
                    alpha=alpha,
                    linewidth=2
                )
            
            # Mark the final position
            plt.scatter(
                trajectory[-1, 0], 
                trajectory[-1, 1], 
                s=100, 
                color=colors[i], 
                edgecolor='black', 
                label=f'Attractor {i+1}'
            )
        
        plt.title(f'Waddington Landscape Attractor Evolution (PCA)')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f"attractor_evolution_epoch_{epoch}.png"))
        if is_final:
            plt.savefig(os.path.join(output_dir, "final_attractor_evolution.png"))
        plt.close()

def analyze_validation_landscape(
    waddington_energy,
    val_loader,
    device,
    output_dir,
    is_multi_cell=False,
    max_samples=1000
):
    """
    Analyze the Waddington landscape using validation data
    """
    waddington_energy.eval()
    
    # Collect gene expressions and calculate energies
    gene_expressions = []
    energy_values = []
    sample_ids = []
    
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing validation landscape"):
            if sample_count >= max_samples:
                break
                
            # Get gene expressions
            gene_expr = batch['gene_expr'].to(device)
            
            # Get IDs depending on dataset type
            if is_multi_cell:
                batch_ids = batch['patch_id']
            else:
                batch_ids = batch['cell_id']
            
            # Calculate energies
            energy = waddington_energy(gene_expr)
            
            # Store data
            if is_multi_cell and len(gene_expr.shape) == 3:
                # Use average gene expression per patch for visualization
                gene_expr = gene_expr.mean(dim=1)
            
            gene_expressions.append(gene_expr.cpu())
            energy_values.append(energy.cpu())
            sample_ids.extend(batch_ids)
            
            sample_count += gene_expr.shape[0]
    
    # Combine data
    gene_expressions = torch.cat(gene_expressions, dim=0).numpy()
    energy_values = torch.cat(energy_values, dim=0).numpy()
    
    # Calculate statistics
    mean_energy = np.mean(energy_values)
    std_energy = np.std(energy_values)
    min_energy = np.min(energy_values)
    max_energy = np.max(energy_values)
    
    logger.info(f"Waddington landscape statistics on validation data:")
    logger.info(f"  Mean energy: {mean_energy:.4f}")
    logger.info(f"  Std dev: {std_energy:.4f}")
    logger.info(f"  Range: [{min_energy:.4f}, {max_energy:.4f}]")
    
    # Perform dimensionality reduction for visualization
    if gene_expressions.shape[1] > 2:
        # PCA first to speed up t-SNE
        pca = PCA(n_components=min(50, gene_expressions.shape[1]))
        gene_expressions_pca = pca.fit_transform(gene_expressions)
        
        # t-SNE for non-linear visualization
        tsne = TSNE(n_components=2, random_state=42)
        gene_expressions_2d = tsne.fit_transform(gene_expressions_pca)
        
        # Plot energy landscape
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with energy-based coloring
        scatter = plt.scatter(
            gene_expressions_2d[:, 0],
            gene_expressions_2d[:, 1],
            c=energy_values,
            cmap='viridis_r',  # Reversed so blue=low energy, yellow=high energy
            alpha=0.7,
            s=50,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Waddington Energy')
        
        plt.title('Waddington Energy Landscape (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "validation_energy_landscape.png"))
        plt.close()
        
        # Save data for further analysis
        landscape_data = {
            'sample_id': sample_ids,
            'tsne_1': gene_expressions_2d[:, 0],
            'tsne_2': gene_expressions_2d[:, 1],
            'energy': energy_values
        }
        
        pd.DataFrame(landscape_data).to_csv(
            os.path.join(output_dir, "validation_landscape_data.csv"),
            index=False
        )
    
    return {
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'min_energy': min_energy,
        'max_energy': max_energy,
        'energy_values': energy_values,
        'sample_ids': sample_ids
    }