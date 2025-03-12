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
from baseline.diffusion import GaussianDiffusion
from src.utils import setup_parser, parse_adata
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from baseline.diffusion_train import generate_images_with_diffusion
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match # Added import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate images using pretrained RNA to H&E model with Diffusion.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths (for reference).')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths (for reference).')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_diffusion_generated', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation (adjust based on GPU memory).') # Smaller default for generation
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels.')
    parser.add_argument('--gen_steps', type=int, default=300, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single',help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels during dataset loading.')
    parser.add_argument('--diffusion_timesteps', type=int, default=300, help='Number of timesteps for diffusion process')
    parser.add_argument('--beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Noise schedule for diffusion')
    parser.add_argument('--predict_noise', action='store_true', default=True, help='Whether model predicts noise (True) or x_0 (False)')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method for diffusion generation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    generated_images_subdir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(generated_images_subdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    missing_gene_symbols_list = None
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        logger.info(f"Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")

    # --- Model Loading from Checkpoint ---
    logger.info(f"Loading pretrained model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return
        
    model_state = checkpoint.get("model", checkpoint)
    model_config_ckpt = checkpoint.get("config", {})

    # Prioritize config from checkpoint for critical architectural params
    args.img_channels = model_config_ckpt.get('img_channels', args.img_channels)
    args.img_size = model_config_ckpt.get('img_size', args.img_size) # Model must match size it was trained on
    # args.model_type = model_config_ckpt.get('model_type', args.model_type) # Decide if cmd line or ckpt takes precedence

    gene_dim = expr_df.shape[1]
    model_constructor_args = dict(
        rna_dim=gene_dim, 
        img_channels=args.img_channels, 
        img_size=args.img_size,
        model_channels=model_config_ckpt.get('model_channels', 128),
        num_res_blocks=model_config_ckpt.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_config_ckpt.get('attention_resolutions', [16,])),
        dropout=model_config_ckpt.get('dropout', 0.1), 
        channel_mult=tuple(model_config_ckpt.get('channel_mult', [1, 2, 2, 2])), 
        use_checkpoint=model_config_ckpt.get('use_checkpoint', False),
        num_heads=model_config_ckpt.get('num_heads', 2), 
        num_head_channels=model_config_ckpt.get('num_head_channels', 16), 
        use_scale_shift_norm=model_config_ckpt.get('use_scale_shift_norm', True),
        resblock_updown=model_config_ckpt.get('resblock_updown', True), 
        use_new_attention_order=model_config_ckpt.get('use_new_attention_order', True), 
        concat_mask=model_config_ckpt.get('concat_mask', getattr(args, 'concat_mask', False)),
        relation_rank=model_config_ckpt.get('relation_rank', getattr(args, 'relation_rank', 50))
    )
    
    model = None
    if args.model_type == 'single': # Use model_type from args (or potentially from ckpt if you prioritize that)
        logger.info("Initializing single-cell model")
        model = RNAtoHnEModel(**model_constructor_args)
    else: 
        logger.info("Initializing multi-cell model")
        model_constructor_args['num_aggregation_heads'] = model_config_ckpt.get('num_aggregation_heads', getattr(args, 'num_aggregation_heads', 4))
        model = MultiCellRNAtoHnEModel(**model_constructor_args)
    
    model.load_state_dict(model_state)
    logger.info(f"Model loaded successfully with img_channels={args.img_channels}, img_size={args.img_size}.")
    model.to(device)
    model.eval()
    # --- End Model Loading ---

    # Create dataset for visualization/generation
    # This dataset primarily provides gene expression and (optionally) real images for comparison
    if args.model_type == 'single':
        image_paths_dict = {}
        if args.image_paths:
            logger.info(f"Loading image paths for reference from {args.image_paths}")
            with open(args.image_paths, "r") as f: image_paths_data = json.load(f)
            image_paths_dict = {k: v for k, v in image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(image_paths_dict)} valid reference cell image paths")
        
        full_dataset = CellImageGeneDataset(
            expr_df, image_paths_dict, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols_list, normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell
        patch_image_paths_dict = None
        with open(args.patch_cell_mapping, "r") as f: patch_to_cells = json.load(f)
        if args.patch_image_paths:
            logger.info(f"Loading patch image paths for reference from {args.patch_image_paths}")
            with open(args.patch_image_paths, "r") as f: patch_image_paths_data = json.load(f)
            patch_image_paths_dict = {k: v for k, v in patch_image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(patch_image_paths_dict)} valid reference patch image paths")
        
        full_dataset = PatchImageGeneDataset(
            expr_df=expr_df, patch_image_paths=patch_image_paths_dict, patch_to_cells=patch_to_cells,
            img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            normalize_aux=args.normalize_aux,
        )
    
    if len(full_dataset) == 0:
        logger.error("Dataset for generation is empty. Cannot proceed.")
        return
    logger.info(f"Full dataset for generation/visualization created with {len(full_dataset)} samples")
    
    num_vis_samples = min(args.num_samples, len(full_dataset))
    if num_vis_samples == 0 and len(dataset) > 0: # Ensure we have at least one sample if dataset is not empty
        num_vis_samples = 1
    if num_vis_samples == 0 :
        logger.warning("Number of samples to visualize is 0. Exiting generation.")
        return
        
    # Take a subset for generation, can be random or first N
    vis_indices = torch.randperm(len(full_dataset))[:num_vis_samples].tolist()
    vis_dataset = torch.utils.data.Subset(full_dataset, vis_indices)
    
    logger.info(f"Will generate images for {len(vis_dataset)} samples.")

    collate_fn_to_use = patch_collate_fn if args.model_type == 'multi' else None
    vis_loader = DataLoader(
        vis_dataset, batch_size=min(args.batch_size, len(vis_dataset)), # Adjust batch size if smaller than num_vis_samples
        shuffle=False, collate_fn=collate_fn_to_use
    )
    
    diffusion = GaussianDiffusion(
        timesteps=args.diffusion_timesteps, beta_schedule=args.beta_schedule,
        predict_noise=args.predict_noise, device=device
    )
    logger.info(f"Initialized diffusion model with {args.diffusion_timesteps} timesteps, {args.beta_schedule} schedule.")
    
    # Get the first (and likely only) batch from vis_loader
    batch = next(iter(vis_loader))
    real_images_tensor = batch['image'].to(device) # For stain norm reference and plotting
    current_batch_size = real_images_tensor.shape[0]
    
    generated_images_tensor = None
    if args.model_type == 'single':
        gene_expr_batch = batch['gene_expr'].to(device)
        sample_ids_in_batch = batch['cell_id']
        gene_mask_batch = batch.get('gene_mask', None)
        if gene_mask_batch is not None: gene_mask_batch = gene_mask_batch.to(device)
        
        logger.info(f"Generating images for {current_batch_size} single-cell samples using diffusion ({args.sampling_method})...")
        with torch.no_grad():
            generated_images_tensor = generate_images_with_diffusion(
                model=model, diffusion=diffusion, gene_expr=gene_expr_batch, device=device,
                num_steps=args.gen_steps, gene_mask=gene_mask_batch, is_multi_cell=False, method=args.sampling_method
            )
    else:  # multi-cell
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr_batch = processed_batch['gene_expr']
        num_cells_info = processed_batch.get('num_cells')
        sample_ids_in_batch = batch['patch_id']
        gene_mask_batch = batch.get('gene_mask', None) # Assuming gene_mask can also be part of multi-cell batch
        if gene_mask_batch is not None: gene_mask_batch = gene_mask_batch.to(device)

        logger.info(f"Generating images for {current_batch_size} multi-cell patches using diffusion ({args.sampling_method})...")
        with torch.no_grad():
            generated_images_tensor = generate_images_with_diffusion(
                model=model, diffusion=diffusion, gene_expr=gene_expr_batch, device=device,
                num_steps=args.gen_steps, num_cells=num_cells_info, gene_mask=gene_mask_batch, is_multi_cell=True, method=args.sampling_method
            )
    logger.info("Image generation complete.")

    # <<< START STAIN NORMALIZATION MODIFICATION >>>
    if args.enable_stain_normalization and args.img_channels >= 3:
        logger.info(f"Applying stain normalization using method: {args.stain_normalization_method}")
        normalized_generated_images_list = []
        for j in range(current_batch_size):
            # Ensure real_images_tensor has corresponding samples if it's shorter (should not happen with current vis_loader setup)
            real_img_np_j = real_images_tensor[j].cpu().numpy().transpose(1, 2, 0) # H,W,C
            gen_img_np_j = generated_images_tensor[j].cpu().numpy().transpose(1, 2, 0) # H,W,C

            real_rgb_j = real_img_np_j[:, :, :3]
            gen_rgb_original_j = gen_img_np_j[:, :, :3]

            if args.stain_normalization_method == 'skimage_hist_match':
                gen_rgb_normalized_j = normalize_staining_rgb_skimage_hist_match(
                    gen_rgb_original_j, real_rgb_j # Use real image as target
                )
            else:
                logger.warning(f"Unsupported stain norm method: {args.stain_normalization_method}. Using original generated image.")
                gen_rgb_normalized_j = gen_rgb_original_j
            
            if args.img_channels > 3: # If there are auxiliary channels
                gen_aux_j = gen_img_np_j[:, :, 3:]
                final_gen_img_np_j = np.concatenate((gen_rgb_normalized_j, gen_aux_j), axis=2)
            else:
                final_gen_img_np_j = gen_rgb_normalized_j
            
            normalized_generated_images_list.append(
                torch.from_numpy(final_gen_img_np_j.transpose(2,0,1)).to(device)
            )
        generated_images_tensor = torch.stack(normalized_generated_images_list) # B,C,H,W
        logger.info("Stain normalization applied to generated images.")
    # <<< END STAIN NORMALIZATION MODIFICATION >>>

    generated_images_tensor = torch.clamp(generated_images_tensor, 0, 1)
    real_images_tensor = torch.clamp(real_images_tensor, 0, 1) # Clamp real images too for consistency

    # Plotting and Saving
    num_plot_samples = current_batch_size # Should be <= num_vis_samples and args.batch_size
    num_extra_channels = max(0, args.img_channels - 3)
    # Rows: Real RGB, Gen RGB, [Real Aux_i, Gen Aux_i] * num_extra_channels
    num_rows_plot = 2 + (2 * num_extra_channels) if real_images_tensor is not None and len(real_images_tensor) > 0 else 1 + (1*num_extra_channels)


    fig, axes = plt.subplots(num_rows_plot, num_plot_samples, figsize=(3 * num_plot_samples, 2.5 * num_rows_plot))
    if num_plot_samples == 1 and num_rows_plot == 1: axes = np.array([[axes]]) # Make it 2D
    elif num_plot_samples == 1: axes = axes.reshape(num_rows_plot, 1)
    elif num_rows_plot == 1: axes = axes.reshape(1, num_plot_samples)


    output_prefix = args.output_name_prefix if args.output_name_prefix else ""
    if output_prefix and not output_prefix.endswith("_"):
        output_prefix += "_"

    for i in range(num_plot_samples):
        current_sample_id = sample_ids_in_batch[i] if i < len(sample_ids_in_batch) else f"sample_{i}"
        
        gen_img_np = generated_images_tensor[i].cpu().numpy().transpose(1, 2, 0)
        
        plot_row_idx = 0
        if real_images_tensor is not None and i < len(real_images_tensor):
            real_img_np = real_images_tensor[i].cpu().numpy().transpose(1, 2, 0)
            axes[plot_row_idx, i].imshow(np.clip(real_img_np[:, :, :3],0,1))
            axes[plot_row_idx, i].set_title(f"Real RGB: {current_sample_id[:10]}") # Shorten ID if too long
            axes[plot_row_idx, i].axis('off')
            plt.imsave(os.path.join(generated_images_subdir, f"{output_prefix}{current_sample_id}_real_rgb.png"), np.clip(real_img_np[:, :, :3],0,1))
            plot_row_idx += 1

        axes[plot_row_idx, i].imshow(np.clip(gen_img_np[:, :, :3],0,1))
        title_suffix = " (norm)" if args.enable_stain_normalization and args.img_channels >=3 else ""
        axes[plot_row_idx, i].set_title(f"Gen RGB{title_suffix}")
        axes[plot_row_idx, i].axis('off')
        plt.imsave(os.path.join(generated_images_subdir, f"{output_prefix}{current_sample_id}_generated_rgb.png"), np.clip(gen_img_np[:, :, :3],0,1))
        plot_row_idx += 1
        
        for c_idx in range(num_extra_channels):
            channel_num = 3 + c_idx
            if real_images_tensor is not None and i < len(real_images_tensor) and real_img_np.shape[2] > channel_num:
                axes[plot_row_idx, i].imshow(np.clip(real_img_np[:, :, channel_num],0,1), cmap='gray')
                axes[plot_row_idx, i].set_title(f"Real Aux {c_idx+1}")
                axes[plot_row_idx, i].axis('off')
                plt.imsave(os.path.join(generated_images_subdir, f"{output_prefix}{current_sample_id}_real_aux{c_idx+1}.png"), np.clip(real_img_np[:, :, channel_num],0,1), cmap='gray')
                plot_row_idx += 1
            
            if gen_img_np.shape[2] > channel_num:
                axes[plot_row_idx, i].imshow(np.clip(gen_img_np[:, :, channel_num],0,1), cmap='gray')
                axes[plot_row_idx, i].set_title(f"Gen Aux {c_idx+1}")
                axes[plot_row_idx, i].axis('off')
                plt.imsave(os.path.join(generated_images_subdir, f"{output_prefix}{current_sample_id}_generated_aux{c_idx+1}.png"), np.clip(gen_img_np[:, :, channel_num],0,1), cmap='gray')
                plot_row_idx += 1
            else: # if generated image doesn't have the aux channel but real does, fill with blank
                 if real_images_tensor is not None and i < len(real_images_tensor) and real_img_np.shape[2] > channel_num :
                    axes[plot_row_idx, i].imshow(np.zeros_like(real_img_np[:,:,0]), cmap='gray') # placeholder
                    axes[plot_row_idx, i].set_title(f"Gen Aux {c_idx+1} (N/A)")
                    axes[plot_row_idx, i].axis('off')
                    plot_row_idx +=1


    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"{output_prefix}diffusion_generation_results.png"))
    plt.close(fig)
    
    logger.info(f"Generated images and summary plot saved to {args.output_dir}")
    if args.enable_stain_normalization and args.img_channels >=3:
        logger.info(f"Stain normalization was enabled using method: {args.stain_normalization_method}")
    else:
        logger.info("Stain normalization was not enabled or not applicable.")

if __name__ == "__main__":
    main()
