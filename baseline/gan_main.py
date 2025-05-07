import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from baseline.gan import CellGAN
from baseline.gan_train import train_with_gan, generate_images_with_gan
from src.utils import setup_parser, parse_adata

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
    parser = argparse.ArgumentParser(description="Train and evaluate RNA to H&E cell image generator with GAN.")
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_gan', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Learning rate for generator optimizer.')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='Learning rate for discriminator optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 Greyscale).')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for training.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--d_steps', type=int, default=1, help='Number of discriminator steps per iteration.')
    parser.add_argument('--g_steps', type=int, default=1, help='Number of generator steps per iteration.')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of the latent noise vector.')
    parser.add_argument('--seed', type=int, default=np.random.randint(100), help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi', 
                        help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--use_wgan', action='store_true', help='Use Wasserstein GAN with gradient penalty.')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight for adversarial loss.')
    parser.add_argument('--l1_weight', type=float, default=10.0, help='Weight for L1 loss.')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for gradient penalty (WGAN-GP).')
    parser.add_argument('--only_inference', action='store_true', help='Skip training and only run inference.')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per RNA profile.')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
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
    
    # Load gene expression data
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
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
        print(len(image_paths))
        
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
    
    if args.model_type == 'multi':
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
    else:
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
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Initialize appropriate generator model
    gene_dim = expr_df.shape[1]  # Number of genes
    
    if args.model_type == 'single':
        logger.info("Initializing single-cell generator")
        generator = RNAtoHnEModel(
            rna_dim=gene_dim,
            img_channels=args.img_channels,
            img_size=args.img_size,
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
            concat_mask=args.concat_mask if hasattr(args, 'concat_mask') else False,
        )
    else:  # multi-cell model
        logger.info("Initializing multi-cell generator")
        generator = MultiCellRNAtoHnEModel(
            rna_dim=gene_dim,
            img_channels=args.img_channels,
            img_size=args.img_size,
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
            concat_mask=args.concat_mask if hasattr(args, 'concat_mask') else False,
        )
    
    logger.info(f"Generator initialized with gene dimension: {gene_dim}")
    
    # Initialize the GAN
    gan = CellGAN(
        generator=generator,
        is_multi_cell=(args.model_type == 'multi'),
        adv_weight=args.adv_weight,
        l1_weight=args.l1_weight,
        gp_weight=args.gp_weight,
        use_wgan=args.use_wgan,
        z_dim=args.z_dim
    )
    logger.info(f"Initialized CellGAN with {'WGAN-GP' if args.use_wgan else 'Standard GAN'} approach")
    
    # Train model
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_rna_to_hne_gan_model.pt")
    if not args.only_inference:
        train_gen_losses, train_disc_losses, val_losses = train_with_gan(
            gan=gan,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            best_model_path=best_model_path,
            patience=args.patience,
            use_amp=args.use_amp,
            weight_decay=args.weight_decay,
            d_steps=args.d_steps,
            g_steps=args.g_steps,
            is_multi_cell=(args.model_type == 'multi')
        )
        logger.info(f"Training complete. Best model saved at {best_model_path}")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_gen_losses, label='Generator Loss')
        plt.plot(train_disc_losses, label='Discriminator Loss')
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
    checkpoint = torch.load(best_model_path, map_location=device)
    gan.generator.load_state_dict(checkpoint["generator"])
    gan.discriminator.load_state_dict(checkpoint["discriminator"])
    gan.generator.to(device)
    gan.discriminator.to(device)
    
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
        
        # Generate images with GAN
        generated_images = generate_images_with_gan(
            gan=gan,
            gene_expr=gene_expr,
            device=device,
            gene_mask=gene_mask,
            num_samples=args.num_samples
        )
    else:  # multi-cell model
        # Prepare batch for multi-cell model
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr = processed_batch['gene_expr']
        num_cells = processed_batch['num_cells']
        real_images = batch['image']
        patch_ids = batch['patch_id']
        
        # Generate images with GAN
        generated_images = generate_images_with_gan(
            gan=gan,
            gene_expr=gene_expr,
            device=device,
            num_cells=num_cells,
            num_samples=args.num_samples
        )

    # Save results
    os.makedirs(os.path.join(args.output_dir, "generated_images"), exist_ok=True)

    # Handle multiple samples if generated
    if args.num_samples > 1:
        # We'll visualize the first sample for each input
        generated_images_to_show = generated_images[:, 0]
        
        # Save all samples
        for i in range(num_vis_samples):
            for s in range(args.num_samples):
                sample_img = generated_images[i, s].cpu().numpy().transpose(1, 2, 0)
                display_id = cell_ids[i] if args.model_type == 'single' else patch_ids[i]
                
                # Save RGB composite
                plt.imsave(
                    os.path.join(args.output_dir, "generated_images", f"{display_id}_gen_sample{s+1}_rgb.png"),
                    sample_img[:,:,:3]
                )
                
                # Save extra channels
                for c in range(3, args.img_channels):
                    plt.imsave(
                        os.path.join(args.output_dir, "generated_images", f"{display_id}_gen_sample{s+1}_ch{c}.png"),
                        sample_img[:,:,c],
                        cmap='gray'
                    )
    else:
        generated_images_to_show = generated_images

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
        gen_img = generated_images_to_show[i].cpu().numpy().transpose(1, 2, 0)
        
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
    
    # Compare multiple samples if generated
    if args.num_samples > 1 and num_vis_samples > 0:
        # Create a figure to show sample diversity
        sample_fig, sample_axes = plt.subplots(
            args.num_samples, num_vis_samples, 
            figsize=(3*num_vis_samples, 3*args.num_samples)
        )
        
        # Ensure sample_axes is a 2D array
        if args.num_samples == 1:
            sample_axes = np.expand_dims(sample_axes, axis=0)
        if num_vis_samples == 1:
            sample_axes = np.expand_dims(sample_axes, axis=1)
            
        # Plot different samples for each input
        for i in range(num_vis_samples):
            for s in range(args.num_samples):
                sample_img = generated_images[i, s].cpu().numpy().transpose(1, 2, 0)
                sample_axes[s, i].imshow(sample_img[:,:,:3])
                if s == 0:
                    sample_axes[s, i].set_title(f"{display_ids[i]}")
                sample_axes[s, i].axis('off')
                
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sample_diversity.png"))
        logger.info(f"Sample diversity visualization saved")

if __name__ == "__main__":
    main()