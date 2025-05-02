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
from dataset import CellImageGeneDataset
from model import RNAtoHnEModel
from rectified_flow import RectifiedFlow, EulerSolver
from rectified_train import train_with_rectified_flow, generate_images_with_rectified_flow

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
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_rectified', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 Greyscale).')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for training.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--gen_steps', type=int, default=100, help='Number of steps for Euler solver during generation.')
    parser.add_argument('--seed', type=int, default=np.random.randint(100), help='Random seed for reproducibility.')
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
    logger.info(f"Loading gene expression data from {args.gene_expr}")
    expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
    
    # Load image paths
    logger.info(f"Loading image paths from {args.image_paths}")
    with open(args.image_paths, "r") as f:
        image_paths = json.load(f)
    logger.info(f"Loaded {len(image_paths)} cell image paths")
    
    # Create dataset
    dataset = CellImageGeneDataset(
        expr_df, 
        image_paths, 
        img_size=args.img_size,
        img_channels=args.img_channels,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.img_size, args.img_size), antialias=True),
        ])
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
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
    
    # Initialize model
    gene_dim = expr_df.shape[1]  # Number of genes
    model = RNAtoHnEModel(
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
    )
    
    logger.info(f"Model initialized with gene dimension: {gene_dim}")
    
    # Initialize the rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")
    
    # Train model
    best_model_path = os.path.join(args.output_dir, "best_rna_to_hne_model_rectified.pt")
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
        weight_decay=args.weight_decay
    )
    
    logger.info(f"Training complete. Best model saved at {best_model_path}")
    
    # Load best model for evaluation
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    
    # Generate images from validation set
    logger.info("Generating images from validation set...")
    
    # Create a subset of the validation set for visualization
    num_vis_samples = min(10, len(val_dataset))
    vis_indices = torch.randperm(len(val_dataset))[:num_vis_samples]
    vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)
    vis_loader = DataLoader(vis_dataset, batch_size=num_vis_samples, shuffle=False)

    # Get a batch of data
    batch = next(iter(vis_loader))
    gene_expr = batch['gene_expr'].to(device)
    real_images = batch['image']
    cell_ids = batch['cell_id']

    # Generate images with rectified flow
    generated_images = generate_images_with_rectified_flow(
        model=model,
        rectified_flow=rectified_flow,
        gene_expr=gene_expr,
        device=device,
        num_steps=args.gen_steps
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

    for i in range(num_vis_samples):
        # Real image processing
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Generated image processing
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Display RGB composites for both real and generated images (first 3 channels)
        # Real RGB composite
        axes[0, i].imshow(real_img[:,:,:3])
        axes[0, i].set_title(f"Real RGB: {cell_ids[i]}")
        axes[0, i].axis('off')
        
        # Generated RGB composite
        axes[1, i].imshow(gen_img[:,:,:3])
        axes[1, i].set_title("Generated RGB")
        axes[1, i].axis('off')
        
        # Save RGB representations
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{cell_ids[i]}_real_rgb.png"),
            real_img[:,:,:3]
        )
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{cell_ids[i]}_gen_rgb.png"),
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
                os.path.join(args.output_dir, "generated_images", f"{cell_ids[i]}_real_ch{c}.png"),
                real_img[:,:,c],
                cmap='gray'
            )
            plt.imsave(
                os.path.join(args.output_dir, "generated_images", f"{cell_ids[i]}_gen_ch{c}.png"),
                gen_img[:,:,c],
                cmap='gray'
            )

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "generation_results.png"))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()