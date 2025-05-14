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
from src.utils import setup_parser, parse_adata
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from waddington.waddington_energy import WaddingtonEnergy
from waddington.waddington_rectified_flow import WaddingtonRectifiedFlow
from waddington.waddington_train import generate_images_with_waddington
from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate images using pretrained RNA to H&E model with Waddington landscape guidance.")
    parser.add_argument('--model_path', type=str, default="cell_256_aux/output_waddington/best_single_waddington_model.pt", help='Path to the pretrained model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/input/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_waddington', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 for auxiliary).')
    parser.add_argument('--gen_steps', type=int, default=100, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    parser.add_argument('--guidance_strength', type=float, default=1.0, help='Strength of Waddington energy guidance during generation.')
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "generated_images"), exist_ok=True)

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

        # Load image paths if provided (for visualization)
        if args.image_paths:
            logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f:
                image_paths = json.load(f)
            logger.info(f"Loaded {len(image_paths)} cell image paths")

            # Filter out non-existent files
            image_paths_tmp = {}
            for k,v in image_paths.items():
                if os.path.exists(v):
                    image_paths_tmp[k] = v
            
            image_paths = image_paths_tmp
            logger.info(f"After filtering: {len(image_paths)} valid cell image paths")
        else:
            image_paths = {}
        
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

            # Filter out non-existent files
            patch_image_paths_tmp = {}
            for k,v in patch_image_paths.items():
                if os.path.exists(v):
                    patch_image_paths_tmp[k] = v
            patch_image_paths = patch_image_paths_tmp
            logger.info(f"After filtering: {len(patch_image_paths)} valid patch image paths")
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
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Create a subset of the dataset for generating images
    num_vis_samples = min(args.num_samples, len(dataset))
    vis_indices = torch.randperm(len(dataset))[:num_vis_samples]
    vis_dataset = torch.utils.data.Subset(dataset, vis_indices)

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
    
    # Initialize appropriate model based on model type
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
    )

    # Load the pretrained model
    logger.info(f"Loading pretrained model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Try loading with the standard model first
    model = None
    try:
        if args.model_type == 'single':
            logger.info("Initializing single-cell model")
            model = RNAtoHnEModel(**model_constructor_args)
        else:  # multi-cell model
            logger.info("Initializing multi-cell model")
            model = MultiCellRNAtoHnEModel(**model_constructor_args, relation_rank=args.relation_rank)
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}")
        
    # If standard model loading failed, try with the deprecated model
    if model is None:
        if args.model_type == 'single':
            logger.info("Initializing single-cell model (deprecated)")
            model = RNAtoHnEModel_deprecation(**model_constructor_args)
        else:  # multi-cell model
            logger.info("Initializing multi-cell model (deprecated)")
            model = MultiCellRNAtoHnEModel_deprecation(**model_constructor_args, relation_rank=args.relation_rank)
        model.load_state_dict(checkpoint["model"])
    
    logger.info(f"Model loaded successfully")
    
    # Load the model to device
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Initialize the Waddington energy model
    logger.info("Initializing Waddington energy model")
    
    # First check the channel count in the checkpoint's energy model
    checkpoint_channels = None
    for key in checkpoint["waddington_energy"]:
        if "image_net.1.weight" in key:
            checkpoint_channels = checkpoint["waddington_energy"][key].shape[1]
            break

    if checkpoint_channels is None:
        checkpoint_channels = args.img_channels

    logger.info(f"Detected {checkpoint_channels} channels in Waddington energy checkpoint (current setting: {args.img_channels})")

    # Create energy model with the correct channel count from checkpoint
    energy_params = {
        'gene_dim': gene_dim,
        'hidden_dim': checkpoint.get('energy_hidden_dim', 128),
        'energy_scale': checkpoint.get('energy_scale', 1.0),
        'num_attractor_states': checkpoint.get('num_attractor_states', 5),
        'img_size': args.img_size,
        'img_channels': checkpoint_channels  # Use channels from checkpoint instead of args
    }

    waddington_energy = WaddingtonEnergy(**energy_params)

    # Now we can load the state dict without channel mismatch errors
    if "waddington_energy" in checkpoint:
        waddington_energy.load_state_dict(checkpoint["waddington_energy"])
        logger.info("Waddington energy model loaded from checkpoint")
    else:
        logger.warning("No Waddington energy model found in checkpoint")
    
    waddington_energy.to(device)
    waddington_energy.eval()
    
    # Initialize the Waddington-guided rectified flow
    waddington_flow = WaddingtonRectifiedFlow(
        sigma_min=0.002, 
        sigma_max=80.0, 
        energy_weight=checkpoint.get('energy_weight', 0.1)
    )
    
    # Set the energy model in the flow
    waddington_flow.waddington_energy = waddington_energy
    
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
        
        # Generate images with Waddington guided flow
        logger.info(f"Generating images for {len(gene_expr)} samples using Waddington guidance...")
        with torch.no_grad():
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
        logger.info("Image generation complete")
    else:  # multi-cell model
        # Prepare batch for multi-cell model
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr = processed_batch['gene_expr']
        num_cells = processed_batch['num_cells']
        real_images = batch['image']
        patch_ids = batch['patch_id']
        
        # Generate images with Waddington guided flow
        logger.info(f"Generating images for {len(real_images)} patches using Waddington guidance...")
        with torch.no_grad():
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
        logger.info("Image generation complete")

    # Calculate energies for visualization
    with torch.no_grad():
        energies = waddington_energy(gene_expr).cpu().numpy()

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
        # Real RGB composite with energy value
        axes[0, i].imshow(real_img[:,:,:3])
        energy_text = f"{display_ids[i]}\nEnergy: {energies[i]:.2f}" if i < len(energies) else display_ids[i]
        axes[0, i].set_title(f"Real RGB: {energy_text}")
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
    
    # Visualize energy distribution
    plt.figure(figsize=(8, 6))
    plt.hist(energies, bins=20, alpha=0.7)
    plt.axvline(np.mean(energies), color='r', linestyle='dashed', linewidth=2)
    plt.title('Distribution of Waddington Energy Values')
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "energy_distribution.png"))
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()