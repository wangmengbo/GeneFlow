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
from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata, analyze_gene_importance
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn, load_hest1k_singlecell_data
from rectified.rectified_train import train_with_rectified_flow, generate_images_with_rectified_flow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# Main Function
# ======================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RNA to H&E cell image generator with Rectified Flow.")
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_rectified', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
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
    parser.add_argument('--hest1k_sid', type=str, default=None, help='HEST-1k sample ID for direct loading')
    parser.add_argument('--hest1k_base_dir', type=str, default=None, help='Base directory for HEST-1k data')
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.hest1k_sid and args.hest1k_base_dir:
        logger.info(f"Loading pre-processed HEST-1k data for sample {args.hest1k_sid}")
        expr_df, image_paths = load_hest1k_singlecell_data(
            args.hest1k_sid, args.hest1k_base_dir, img_size=args.img_size, img_channels=args.img_channels
        )
        missing_gene_symbols = None
    elif args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
        missing_gene_symbols = None
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
    gene_names = expr_df.columns.tolist()

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")
        if not (args.hest1k_sid and args.hest1k_base_dir):
            logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f:
                image_paths = json.load(f)
            logger.info(f"Loaded {len(image_paths)} cell image paths")
            image_paths_tmp = {}
            for k, v in image_paths.items():
                if os.path.exists(v):
                    image_paths_tmp[k] = v
            image_paths = image_paths_tmp
            logger.info(f"Filtered to {len(image_paths)} existing image paths")
        dataset = CellImageGeneDataset(
            expr_df,
            image_paths,
            img_size=args.img_size,
            img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]),
            missing_gene_symbols=missing_gene_symbols,
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
            print(len(patch_image_paths))

            logger.info(f"Loaded {len(patch_image_paths)} patch image paths")
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
        rna_dim= gene_dim,
        img_channels= args.img_channels,
        img_size= args.img_size,
        model_channels= 128,
        num_res_blocks= 2,
        attention_resolutions= (16,),
        dropout= 0.1,        
        channel_mult= (1, 2, 2, 2),
        use_checkpoint= False,
        num_heads= 2,    
        num_head_channels= 16, 
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        concat_mask= args.concat_mask,
        relation_rank= args.relation_rank,
    )

    if args.model_type == 'single':
        logger.info("Initializing single-cell model")
        model = RNAtoHnEModel(**model_constructor_args)
    else:  # multi-cell model
        logger.info("Initializing multi-cell model")
        model = MultiCellRNAtoHnEModel(**model_constructor_args, num_aggregation_heads=args.num_aggregation_heads)
    
    logger.info(f"Model initialized with gene dimension: {gene_dim}")
    
    # Initialize the rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")
    
    # Train model
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_rna_to_hne_model_rectified.pt")
    if not args.only_inference:
        train_losses, val_losses = train_with_rectified_flow(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            rectified_flow=rectified_flow,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=best_model_path,
            patience=args.patience,
            use_amp=args.use_amp,
            weight_decay=args.weight_decay,
            is_multi_cell=(args.model_type == 'multi')  # Pass flag for multi-cell model
        )
        logger.info(f"Training complete. Best model saved at {best_model_path}")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    else:
        logger.info(f"Skipping training. Using existing model at {best_model_path}")

    # Load best model for evaluation
    logger.info(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    
    # Generate images from validation set
    logger.info("Generating images from validation set...")
    
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
            collate_fn=patch_collate_fn  # Use the same custom collate function
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
        
        # Generate images with rectified flow
        generated_images = generate_images_with_rectified_flow(
            model=model,
            rectified_flow=rectified_flow,
            gene_expr=gene_expr,
            device=device,
            num_steps=args.gen_steps,
            gene_mask=gene_mask,
            is_multi_cell=False
        )
    else:  # multi-cell model
        # Prepare batch for multi-cell model
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr = processed_batch['gene_expr']
        num_cells = processed_batch['num_cells']
        real_images = batch['image']
        patch_ids = batch['patch_id']
        
        # Generate images with rectified flow
        generated_images = generate_images_with_rectified_flow(
            model=model,
            rectified_flow=rectified_flow,
            gene_expr=gene_expr,
            device=device,
            num_steps=args.gen_steps,
            num_cells=num_cells,
            is_multi_cell=True
        )

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

    # Get IDs for display
    display_ids = cell_ids if args.model_type == 'single' else patch_ids

    for i in range(num_vis_samples):
        # Real image processing
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Generated image processing
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Display RGB composites for both real and generated images (first 3 channels)
        # Real RGB composite
        axes[0, i].imshow(real_img[:,:,:3])
        axes[0, i].set_title(f"Real RGB: {display_ids[i]}")
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
    plt.savefig(os.path.join(args.output_dir, "generation_results.png"))
    
    logger.info(f"Results saved to {args.output_dir}")

    # Skip importance analysis for multi-cell model for now
    if args.model_type == 'single':
        importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
        analyze_gene_importance(
            model=model,
            data_loader=val_loader, # Use validation loader
            rectified_flow=rectified_flow,
            device=device,
            gene_names=gene_names, # Pass the list of gene names
            output_path=importance_output_path,
            timesteps_to_analyze=args.analysis_timesteps if hasattr(args, 'analysis_timesteps') else None,
            num_batches_to_analyze=args.analysis_batches if hasattr(args, 'analysis_batches') else None
        )
        logger.info(f"Gene importance scores saved to {importance_output_path}")

    if hasattr(args, 'only_importance_anlaysis') and args.only_importance_anlaysis:
        logger.info("Skipping image generation as only importance analysis is requested.")
        return

if __name__ == "__main__":
    main()