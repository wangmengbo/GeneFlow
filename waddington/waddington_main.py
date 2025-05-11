import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.utils import setup_parser, parse_adata, analyze_gene_importance
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn

# Import Waddington-specific modules
from waddington.waddington_energy import WaddingtonEnergy
from waddington.waddington_rectified_flow import WaddingtonRectifiedFlow
from waddington.waddington_train import train_with_waddington_guidance, generate_images_with_waddington

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# Main Function for Waddington Guidance
# ======================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RNA to H&E cell image generator with Waddington landscape guidance.")
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/input/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_waddington', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 Greyscale).')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for training.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--gen_steps', type=int, default=100, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
    parser.add_argument('--only_inference1', action='store_true', help='Only run inference using an existing model.')
    
    # Waddington-specific arguments
    parser.add_argument('--energy_weight', type=float, default=0.1, help='Weight of Waddington energy term in loss.')
    parser.add_argument('--energy_scale', type=float, default=1.0, help='Scale factor for Waddington energy.')
    parser.add_argument('--num_attractor_states', type=int, default=5, help='Number of attractor states in Waddington model.')
    parser.add_argument('--energy_hidden_dim', type=int, default=128, help='Hidden dimension for Waddington energy network.')
    parser.add_argument('--guidance_strength', type=float, default=1.0, help='Strength of energy guidance during generation.')
    parser.add_argument('--visualize_attractors', action='store_true', help='Create additional visualizations of attractor states.')
    
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    waddington_dir = os.path.join(args.output_dir, "waddington")
    os.makedirs(waddington_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load gene expression data
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        logger.info(f"Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
    gene_names = expr_df.columns.tolist()

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")

        # Load image paths
        logger.info(f"Loading image paths from {args.image_paths}")
        with open(args.image_paths, "r") as f:
            image_paths = json.load(f)
        logger.info(f"Loaded {len(image_paths)} cell image paths")

        image_paths_tmp = {}
        for k,v in image_paths.items():
            if os.path.exists(v):
                image_paths_tmp[k] = v
        
        image_paths = image_paths_tmp
        logger.info(f"After filtering, using {len(image_paths)} valid image paths")
        
        dataset = CellImageGeneDataset(
            expr_df, 
            image_paths, 
            img_size=args.img_size,
            img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]),
            missing_gene_symbols=missing_gene_symbols if 'missing_gene_symbols' in locals() else None,
            normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell model
        logger.info("Creating multi-cell dataset")
        # Load patch-to-cell mapping
        logger.info(f"Loading patch-to-cell mapping from {args.patch_cell_mapping}")
        with open(args.patch_cell_mapping, "r") as f:
            patch_to_cells = json.load(f)
        
        # Load patch image paths if provided
        if args.patch_image_paths:
            logger.info(f"Loading patch image paths from {args.patch_image_paths}")
            with open(args.patch_image_paths, "r") as f:
                patch_image_paths = json.load(f)

            patch_image_paths_tmp = {}
            for k,v in patch_image_paths.items():
                if os.path.exists(v):
                    patch_image_paths_tmp[k] = v
            patch_image_paths = patch_image_paths_tmp
            logger.info(f"After filtering, using {len(patch_image_paths)} valid patch image paths")
        else:
            patch_image_paths = None
            
        dataset = PatchImageGeneDataset(
            expr_df=expr_df,
            patch_image_paths=patch_image_paths,
            patch_to_cells=patch_to_cells,
            img_size=args.img_size,
            img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]),
            normalize_aux=args.normalize_aux,
        )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    if args.model_type == 'single':
        # Use default collate for single-cell dataset
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    else:
        # Use custom collate function for multi-cell dataset
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            collate_fn=patch_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            collate_fn=patch_collate_fn
        )
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Initialize appropriate model
    gene_dim = expr_df.shape[1]  # Number of genes
    
    model_constructor_args = dict(
        rna_dim=gene_dim,
        img_channels=args.img_channels,
        img_size=args.img_size,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,        
        channel_mult=(1, 2, 2, 2),
        use_checkpoint=False,
        num_heads=2,    
        num_head_channels=16, 
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        concat_mask=args.concat_mask,
        relation_rank=args.relation_rank,
    )

    if args.model_type == 'single':
        logger.info("Initializing single-cell model")
        model = RNAtoHnEModel(**model_constructor_args)
    else:  # multi-cell model
        logger.info("Initializing multi-cell model")
        model = MultiCellRNAtoHnEModel(**model_constructor_args, num_aggregation_heads=args.num_aggregation_heads)
    
    logger.info(f"Model initialized with gene dimension: {gene_dim}")
    
    # Initialize the Waddington components
    logger.info("Initializing Waddington landscape energy components")
    
    # Create and configure the Waddington energy model
    waddington_energy = WaddingtonEnergy(
        gene_dim=gene_dim,
        hidden_dim=args.energy_hidden_dim,
        energy_scale=args.energy_scale,
        num_attractor_states=args.num_attractor_states,
        img_size=args.img_size,
        img_channels=args.img_channels
    )
    
    # Create the Waddington-guided rectified flow
    waddington_flow = WaddingtonRectifiedFlow(
        sigma_min=0.002,
        sigma_max=80.0,
        energy_weight=args.energy_weight
    )
    
    # Connect the energy model to the flow
    waddington_flow.waddington_energy = waddington_energy
    
    # Path for the best model
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_waddington_model.pt")
    
    # Train or load model
    if not args.only_inference1:
        logger.info("Starting training with Waddington landscape guidance")
        train_results = train_with_waddington_guidance(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=best_model_path,
            patience=args.patience,
            use_amp=args.use_amp,
            weight_decay=args.weight_decay,
            is_multi_cell=(args.model_type == 'multi'),
            energy_weight=args.energy_weight,
            energy_scale=args.energy_scale,
            num_attractor_states=args.num_attractor_states,
            hidden_dim=args.energy_hidden_dim,
            output_dir=waddington_dir
        )
        
        # Extract training metrics
        train_losses = train_results['train_losses']
        val_losses = train_results['val_losses']
        train_energy_losses = train_results['train_energy_losses']
        val_energy_losses = train_results['val_energy_losses']
        
        # Plot combined training curves
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_energy_losses, label='Train Energy')
        plt.plot(val_energy_losses, label='Validation Energy')
        plt.xlabel('Epoch')
        plt.ylabel('Energy')
        plt.title('Waddington Energy Term')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "waddington_training_curves.png"))
        
        logger.info(f"Training complete. Best model saved at {best_model_path}")
    else:
        logger.info(f"Skipping training. Using existing model at {best_model_path}")

    # Load best model for evaluation
    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)

    # Check for image channel mismatch in the checkpoint
    checkpoint_channels = None
    for key in checkpoint["waddington_energy"]:
        if "image_net.1.weight" in key:
            checkpoint_channels = checkpoint["waddington_energy"][key].shape[1]
            break

    if checkpoint_channels is not None and checkpoint_channels != args.img_channels:
        logger.info(f"Detected image channel mismatch: checkpoint has {checkpoint_channels}, current setting is {args.img_channels}")
        
        # Create a new Waddington energy model with the correct channel count
        waddington_energy = WaddingtonEnergy(
            gene_dim=gene_dim,
            hidden_dim=args.energy_hidden_dim,
            energy_scale=args.energy_scale,
            num_attractor_states=args.num_attractor_states,
            img_size=args.img_size,
            img_channels=checkpoint_channels  # Use channel count from checkpoint
        )
        
        # Now we can load the state dict directly
        waddington_energy.load_state_dict(checkpoint["waddington_energy"])
        
        # Update for future use
        args.img_channels_energy = checkpoint_channels
    else:
        # Load Waddington energy model state directly if no mismatch
        waddington_energy.load_state_dict(checkpoint["waddington_energy"])

    # Move model to the appropriate device
    waddington_energy.to(device)
    
    # Set energy model in flow for generation
    waddington_flow.waddington_energy = waddington_energy
    
    # Generate images from validation set
    logger.info("Generating images from validation set with Waddington guidance...")
    
    # Create a subset of the validation set for visualization
    num_vis_samples = min(10, len(val_dataset))
    vis_indices = torch.randperm(len(val_dataset))[:num_vis_samples]
    vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)

    # Use the appropriate collate function based on model type
    if args.model_type == 'multi':
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=num_vis_samples, 
            shuffle=False,
            collate_fn=patch_collate_fn
        )
    else:
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=num_vis_samples, 
            shuffle=False
        )

    # Get a batch of data
    batch = next(iter(vis_loader))
    
    # Handle data differently based on model type
    if args.model_type == 'single':
        gene_expr = batch['gene_expr'].to(device)
        real_images = batch['image']
        cell_ids = batch['cell_id']
        gene_mask = batch.get('gene_mask', None)
        if gene_mask is not None:
            gene_mask = gene_mask.to(device)
        
        # Generate images with Waddington guidance
        generated_images = generate_images_with_waddington(
            model=model,
            waddington_flow=waddington_flow,
            gene_expr=gene_expr,
            device=device,
            num_steps=args.gen_steps,
            gene_mask=gene_mask,
            is_multi_cell=False,
            guidance_strength=args.guidance_strength
        )
        
        # For display
        display_ids = cell_ids
    else:  # multi-cell model
        # Prepare batch for multi-cell model
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr = processed_batch['gene_expr']
        num_cells = processed_batch['num_cells']
        real_images = batch['image']
        patch_ids = batch['patch_id']
        
        # Generate images with Waddington guidance
        generated_images = generate_images_with_waddington(
            model=model,
            waddington_flow=waddington_flow,
            gene_expr=gene_expr,
            device=device,
            num_steps=args.gen_steps,
            num_cells=num_cells,
            is_multi_cell=True,
            guidance_strength=args.guidance_strength
        )
        
        # For display
        display_ids = patch_ids

    # Save results
    os.makedirs(os.path.join(args.output_dir, "generated_images"), exist_ok=True)

    # Calculate number of extra channels beyond RGB
    num_channels = args.img_channels
    num_extra_channels = max(0, num_channels - 3)

    # Calculate number of rows needed:
    # 2 rows for RGB (real and generated)
    # Plus 2 rows for each extra channel (real and generated)
    num_rows = 2 + (2 * num_extra_channels)

    # Create the figure
    fig, axes = plt.subplots(num_rows, num_vis_samples, figsize=(3*num_vis_samples, 2*num_rows))

    # Ensure axes is a 2D array for consistent indexing
    if num_vis_samples == 1:
        axes = np.expand_dims(axes, axis=1)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Calculate energy for each sample for visualization
    with torch.no_grad():
        sample_energies = waddington_energy(gene_expr).cpu().numpy()

    for i in range(num_vis_samples):
        # Real image processing
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Generated image processing
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Display RGB composites for both real and generated images (first 3 channels)
        # Real RGB composite
        axes[0, i].imshow(real_img[:,:,:3])
        energy_text = f"Energy: {sample_energies[i]:.2f}" if i < len(sample_energies) else ""
        axes[0, i].set_title(f"Real RGB: {display_ids[i]}\n{energy_text}")
        axes[0, i].axis('off')
        
        # Generated RGB composite
        axes[1, i].imshow(gen_img[:,:,:3])
        axes[1, i].set_title("Generated RGB")
        axes[1, i].axis('off')
        
        # Save RGB representations
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_real_rgb.png"),
            real_img[:,:,:3]
        )
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_gen_rgb.png"),
            gen_img[:,:,:3]
        )
        
        # Display each extra channel separately (channels 3 and beyond)
        for c in range(3, num_channels):
            # Calculate row indices for extra channels
            real_row_idx = 2 + (2 * (c - 3))
            gen_row_idx = 3 + (2 * (c - 3))
            
            # Real image extra channel
            axes[real_row_idx, i].imshow(real_img[:,:,c], cmap='gray')
            axes[real_row_idx, i].set_title(f"Real Ch{c}")
            axes[real_row_idx, i].axis('off')
            
            # Generated image extra channel
            axes[gen_row_idx, i].imshow(gen_img[:,:,c], cmap='gray')
            axes[gen_row_idx, i].set_title(f"Gen Ch{c}")
            axes[gen_row_idx, i].axis('off')
            
            # Save individual extra channel images
            plt.imsave(
                os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_real_ch{c}.png"),
                real_img[:,:,c],
                cmap='gray'
            )
            plt.imsave(
                os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_gen_ch{c}.png"),
                gen_img[:,:,c],
                cmap='gray'
            )

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "waddington_generation_results.png"))
    
    # Visualize attractor states if requested
    if args.visualize_attractors:
        logger.info("Visualizing Waddington attractor states...")
        attractor_states = waddington_energy.get_attractor_states().numpy()
        
        # If gene dimension is large, use PCA for visualization
        if gene_dim > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            attractor_states_2d = pca.fit_transform(attractor_states)
            
            # Plot attractors
            plt.figure(figsize=(10, 8))
            for i in range(args.num_attractor_states):
                plt.scatter(
                    attractor_states_2d[i, 0], 
                    attractor_states_2d[i, 1], 
                    s=200, 
                    label=f'Attractor {i+1}'
                )
            
            plt.title('Waddington Landscape Attractor States (PCA)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.savefig(os.path.join(args.output_dir, "waddington_attractors.png"))
            
            # Save attractor information
            attractor_df = pd.DataFrame(
                attractor_states, 
                columns=gene_names
            )
            attractor_df.to_csv(os.path.join(args.output_dir, "waddington_attractors.csv"))
    
    logger.info(f"Results saved to {args.output_dir}")
    
    # Analyze gene importance for single-cell model
    if args.model_type == 'single':
        logger.info("Analyzing gene importance...")
        importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
        analyze_gene_importance(
            model=model,
            data_loader=val_loader,
            device=device,
            gene_names=gene_names,
            output_path=importance_output_path
        )
        logger.info(f"Gene importance scores saved to {importance_output_path}")
        
        # Also analyze Waddington energy effect on gene expression
        logger.info("Analyzing Waddington energy influence on gene expression...")
        
        # Collect gene expressions and energies from validation set
        gene_expressions = []
        energies = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene_expr_batch = batch['gene_expr'].to(device)
                energy_batch = waddington_energy(gene_expr_batch).cpu().numpy()
                
                gene_expressions.append(batch['gene_expr'].cpu().numpy())
                energies.extend(energy_batch)
                
                if len(gene_expressions) * args.batch_size > 1000:
                    break  # Limit to 1000 samples for efficiency
        
        # Combine data
        gene_expressions = np.vstack(gene_expressions)
        energies = np.array(energies)
        
        # Calculate correlation between gene expression and energy
        correlations = []
        for i in range(gene_dim):
            corr = np.corrcoef(gene_expressions[:, i], energies)[0, 1]
            correlations.append((gene_names[i], corr))
        
        # Sort by absolute correlation and save
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        correlation_df = pd.DataFrame(correlations, columns=['gene', 'energy_correlation'])
        correlation_df.to_csv(os.path.join(args.output_dir, "waddington_gene_correlations.csv"), index=False)
        
        # Plot top 10 most correlated genes
        top_genes = correlation_df.head(10)
        plt.figure(figsize=(12, 6))
        plt.bar(top_genes['gene'], top_genes['energy_correlation'])
        plt.title('Top 10 Genes Correlated with Waddington Energy')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "waddington_top_gene_correlations.png"))

if __name__ == "__main__":
    main()