import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.single_model import RNAtoHnEModel
from baseline.diffusion import GaussianDiffusion
from rectified.rectified_flow import RectifiedFlow
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from baseline.diffusion_train import generate_images_with_rectified_flow, generate_images_with_diffusion
from src.utils import setup_parser, parse_adata, analyze_gene_importance, analyze_gene_importance_diffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA to H&E cell image generator with saved model.")
    parser.add_argument('--model_path', type=str, default='/home/verma198/Public/hne/cell_256_aux/output_rectified/best_multi_rna_to_hne_model_diffusion.pt', help='Path to saved model checkpoint.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/evaluation_output', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for evaluation.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 for auxiliary).')
    parser.add_argument('--gen_steps', type=int, default=100, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--method', type=str, choices=['diffusion', 'rectified_flow'], default='diffusion',help='Generation method: diffusion or rectified flow')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    parser.add_argument('--analysis', action='store_true', help='Perform gene importance analysis.')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of timesteps for diffusion process')
    parser.add_argument('--beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Noise schedule for diffusion')
    parser.add_argument('--predict_noise', action='store_true', help='Whether model predicts noise (True) or x_0 (False)')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default='ddim', help='Sampling method for diffusion generation')
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

        # Filter out non-existent image paths
        image_paths_tmp = {}
        for k, v in image_paths.items():
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

            # Filter out non-existent image paths
            patch_image_paths_tmp = {}
            for k, v in patch_image_paths.items():
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
    
    # Create evaluation dataset (using the full dataset for evaluation)
    eval_dataset = dataset
    
    # Use random subset for visualization
    num_vis_samples = min(args.num_samples, len(eval_dataset))
    vis_indices = torch.randperm(len(eval_dataset))[:num_vis_samples]
    vis_dataset = torch.utils.data.Subset(eval_dataset, vis_indices)

    # Use the appropriate collate function based on model type
    if args.model_type == 'multi':
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=num_vis_samples, 
            shuffle=False,
            collate_fn=patch_collate_fn
        )
        # Also create a dataloader for gene importance analysis if needed
        if args.analysis:
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                collate_fn=patch_collate_fn
            )
    else:
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=num_vis_samples, 
            shuffle=False
        )
        # Also create a dataloader for gene importance analysis if needed
        if args.analysis:
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=args.batch_size, 
                shuffle=False
            )
    
    logger.info(f"Evaluation set size: {len(eval_dataset)}, Visualization samples: {num_vis_samples}")
    
    # Initialize appropriate model
    gene_dim = expr_df.shape[1]  # Number of genes
    
    if args.model_type == 'single':
        logger.info("Initializing single-cell model")
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
            concat_mask=args.concat_mask if hasattr(args, 'concat_mask') else False,
        )
    else:  # multi-cell model
        logger.info("Initializing multi-cell model")
        model = MultiCellRNAtoHnEModel(
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
    
    # Load the saved model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    # Initialize the appropriate flow model
    if args.method == 'rectified_flow':
        logger.info("Initializing rectified flow")
        flow_model = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    else:  # diffusion
        logger.info("Initializing diffusion model")
        flow_model = GaussianDiffusion(
            timesteps=args.diffusion_timesteps,
            beta_schedule=args.beta_schedule,
            predict_noise=args.predict_noise,
            device=device
        )
    
    # Perform gene importance analysis if requested
    if args.analysis and args.model_type == 'single':
        importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
        logger.info(f"Performing gene importance analysis...")
        
        if args.method == 'diffusion':
            analyze_gene_importance_diffusion(
                model=model,
                data_loader=eval_loader,
                diffusion=flow_model,
                device=device,
                gene_names=gene_names,
                output_path=importance_output_path,
                timesteps_to_analyze=args.analysis_timesteps if hasattr(args, 'analysis_timesteps') else None,
                num_batches_to_analyze=args.analysis_batches if hasattr(args, 'analysis_batches') else None
            )
        else:  # rectified_flow
            analyze_gene_importance(
                model=model,
                data_loader=eval_loader,
                rectified_flow=flow_model,
                device=device,
                gene_names=gene_names,
                output_path=importance_output_path,
                num_batches_to_analyze=args.analysis_batches if hasattr(args, 'analysis_batches') else None
            )
        logger.info(f"Gene importance scores saved to {importance_output_path}")
    
    # Get a batch of data for generation
    batch = next(iter(vis_loader))

    # Generate images based on model type and method
    with torch.no_grad():
        if args.model_type == 'single':
            gene_expr = batch['gene_expr'].to(device)
            real_images = batch['image']
            cell_ids = batch['cell_id']
            gene_mask = batch.get('gene_mask', None)
            if gene_mask is not None:
                gene_mask = gene_mask.to(device)
            
            if args.method == 'diffusion':
                logger.info("Generating images with diffusion...")
                generated_images = generate_images_with_diffusion(
                    model=model,
                    diffusion=flow_model,
                    gene_expr=gene_expr,
                    device=device,
                    num_steps=args.gen_steps,
                    gene_mask=gene_mask,
                    is_multi_cell=False,
                    method=args.sampling_method
                )
            else:  # rectified_flow
                logger.info("Generating images with rectified flow...")
                generated_images = generate_images_with_rectified_flow(
                    model=model,
                    rectified_flow=flow_model,
                    gene_expr=gene_expr,
                    device=device,
                    num_steps=args.gen_steps,
                    gene_mask=gene_mask
                )
        else:  # multi-cell model
            # Prepare batch for multi-cell model
            processed_batch = prepare_multicell_batch(batch, device)
            gene_expr = processed_batch['gene_expr']
            num_cells = processed_batch['num_cells']
            real_images = batch['image']
            patch_ids = batch['patch_id']
            
            if args.method == 'diffusion':
                logger.info("Generating images with diffusion...")
                generated_images = generate_images_with_diffusion(
                    model=model,
                    diffusion=flow_model,
                    gene_expr=gene_expr,
                    device=device,
                    num_steps=args.gen_steps,
                    num_cells=num_cells,
                    is_multi_cell=True,
                    method=args.sampling_method
                )
            else:  # rectified_flow
                logger.info("Generating images with rectified flow...")
                generated_images = generate_images_with_rectified_flow(
                    model=model,
                    rectified_flow=flow_model,
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

if __name__ == "__main__":
    main()