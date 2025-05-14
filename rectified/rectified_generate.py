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
from src.utils import setup_parser, parse_adata
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from rectified.rectified_train import generate_images_with_rectified_flow
from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match # Added import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate images using pretrained RNA to H&E model with Rectified Flow.")
    parser.add_argument('--model_path', type=str, default="cell_256_aux/output_rectified/best_single_rna_to_hne_model_rectified-multihead1.pt", help='Path to the pretrained model.')
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
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    # Arguments for stain normalization will be added by setup_parser
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
    missing_gene_symbols_list = None # Initialize
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
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
                image_paths_dict = json.load(f) # Renamed to avoid conflict
            logger.info(f"Loaded {len(image_paths_dict)} cell image paths")

            # Filter out non-existent files
            image_paths_tmp = {}
            for k,v in image_paths_dict.items():
                if os.path.exists(v):
                    image_paths_tmp[k] = v
            
            image_paths_dict = image_paths_tmp
            logger.info(f"After filtering: {len(image_paths_dict)} valid cell image paths")
        else:
            image_paths_dict = {}
        
        dataset = CellImageGeneDataset(
            expr_df, 
            image_paths_dict, 
            img_size=args.img_size,
            img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]),
            missing_gene_symbols=missing_gene_symbols_list, # Use the parsed list
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
                patch_image_paths_dict = json.load(f) # Renamed

            # Filter out non-existent files
            patch_image_paths_tmp = {}
            for k,v in patch_image_paths_dict.items():
                if os.path.exists(v):
                    patch_image_paths_tmp[k] = v
            patch_image_paths_dict = patch_image_paths_tmp
            logger.info(f"After filtering: {len(patch_image_paths_dict)} valid patch image paths")
        else:
            patch_image_paths_dict = None # Keep as None if not provided
            
        dataset = PatchImageGeneDataset(
            expr_df=expr_df,
            patch_image_paths=patch_image_paths_dict,
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
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create a subset of the dataset for generating images
    num_vis_samples = min(args.num_samples, len(dataset))
    if num_vis_samples == 0 and len(dataset) > 0: # Ensure we have at least one sample if dataset is not empty
        num_vis_samples = 1
    elif len(dataset) == 0:
        logger.error("Dataset is empty after splitting. Exiting.")
        return

    vis_indices = torch.randperm(len(dataset))[:num_vis_samples].tolist() # Ensure it's a list
    vis_dataset = torch.utils.data.Subset(dataset, vis_indices)

    # Use the appropriate collate function based on model type
    if args.model_type == 'multi':
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=args.batch_size if num_vis_samples > args.batch_size else num_vis_samples, # Adjust batch size for small num_vis_samples
            shuffle=False,
            collate_fn=patch_collate_fn
        )
    else:
        vis_loader = DataLoader(
            vis_dataset, 
            batch_size=args.batch_size if num_vis_samples > args.batch_size else num_vis_samples, # Adjust batch size
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
        attention_resolutions= (16,), # Ensure tuple
        dropout= 0.1,        
        channel_mult= (1, 2, 2, 2), # Ensure tuple
        use_checkpoint= False,
        num_heads= 2,    
        num_head_channels= 16, 
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        concat_mask= args.concat_mask,
        relation_rank= args.relation_rank if hasattr(args, 'relation_rank') else 50 # Default if not in args
    )

    # Load the pretrained model
    logger.info(f"Loading pretrained model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device) # Changed weights_only to False to load config if present
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return
        
    model_state = checkpoint.get("model", checkpoint) # Handle checkpoints that might just be state_dict
    model_config_ckpt = checkpoint.get("config", {})


    # Update args from model_config_ckpt if they were used for training and differ
    # This helps ensure consistency if the generation script uses different defaults
    # than the training script.
    args.img_channels = model_config_ckpt.get('img_channels', args.img_channels)
    args.img_size = model_config_ckpt.get('img_size', args.img_size)
    # Potentially update other architectural args from model_config_ckpt if necessary
    # For example: args.model_type = model_config_ckpt.get('model_type', args.model_type)
    
    current_model_type = model_config_ckpt.get('model_type', args.model_type) # Prioritize checkpoint config

    model = None
    try:
        if current_model_type == 'single':
            logger.info("Initializing single-cell model")
            model = RNAtoHnEModel(**model_constructor_args)
        else:
            logger.info("Initializing multi-cell model")
            # Ensure num_aggregation_heads is available for multi-cell model
            model_constructor_args['num_aggregation_heads'] = args.num_aggregation_heads if hasattr(args, 'num_aggregation_heads') else 4
            model = MultiCellRNAtoHnEModel(**model_constructor_args)
        model.load_state_dict(model_state)
    except Exception as e:
        logger.warning(f"Failed to load model with current constructor: {e}. Trying deprecated constructor.")
        # Remove args not present in deprecated constructors or adjust them
        deprecated_constructor_args = model_constructor_args.copy()
        if 'relation_rank' in deprecated_constructor_args and current_model_type == 'single':
            deprecated_constructor_args.pop('relation_rank')
        if 'num_aggregation_heads' in deprecated_constructor_args: # This was not in original deprecated
             deprecated_constructor_args.pop('num_aggregation_heads')


        if current_model_type == 'single':
            model = RNAtoHnEModel_deprecation(**deprecated_constructor_args)
        else:
            model = MultiCellRNAtoHnEModel_deprecation(**deprecated_constructor_args)
        model.load_state_dict(model_state)
        
    logger.info(f"Model loaded successfully using {current_model_type} constructor.")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Initialize the rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")
    
    # Get a batch of data
    try:
        batch = next(iter(vis_loader))
    except StopIteration:
        logger.error("Visualization data loader is empty. Cannot generate images.")
        return

    # Handle data differently based on model type
    if current_model_type == 'single':
        gene_expr = batch['gene_expr'].to(device)
        real_images_tensor = batch['image'] # Keep as tensor for now
        display_ids = batch['cell_id']
        gene_mask = batch.get('gene_mask', None)
        if gene_mask is not None:
            gene_mask = gene_mask.to(device)
        
        logger.info(f"Generating images for {len(gene_expr)} single-cell samples using rectified flow...")
        with torch.no_grad():
            generated_images_tensor = generate_images_with_rectified_flow(
                model=model,
                rectified_flow=rectified_flow,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                gene_mask=gene_mask,
                is_multi_cell=False
            )
    else:  # multi-cell model
        processed_batch = prepare_multicell_batch(batch, device)
        gene_expr = processed_batch['gene_expr']
        num_cells_info = processed_batch['num_cells'] # Renamed to avoid conflict
        real_images_tensor = batch['image'] # Keep as tensor
        display_ids = batch['patch_id']
        
        logger.info(f"Generating images for {len(real_images_tensor)} multi-cell patches using rectified flow...")
        with torch.no_grad():
            generated_images_tensor = generate_images_with_rectified_flow(
                model=model,
                rectified_flow=rectified_flow,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                num_cells=num_cells_info,
                is_multi_cell=True
            )
    logger.info("Image generation complete")

    # Actual number of images generated/visualized might be less than num_vis_samples if batch_size was smaller
    actual_vis_count = real_images_tensor.shape[0]

    num_channels = args.img_channels
    num_extra_channels = max(0, num_channels - 3)
    num_rows = 2 + (2 * num_extra_channels)

    fig, axes = plt.subplots(num_rows, actual_vis_count, figsize=(3*actual_vis_count, 2.5*num_rows)) # Adjusted fig height
    if actual_vis_count == 1:
        axes = np.expand_dims(axes, axis=1)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(actual_vis_count):
        real_img_np = real_images_tensor[i].cpu().numpy().transpose(1, 2, 0) # H, W, C
        gen_img_np = generated_images_tensor[i].cpu().numpy().transpose(1, 2, 0) # H, W, C

        current_display_id = display_ids[i] if i < len(display_ids) else f"Sample_{i}"

        # <<< START STAIN NORMALIZATION MODIFICATION >>>
        if args.enable_stain_normalization and num_channels >= 3:
            logger.info(f"Applying stain normalization to generated sample {current_display_id} using method: {args.stain_normalization_method}")
            
            real_rgb_for_norm = real_img_np[:, :, :3]
            gen_rgb_original = gen_img_np[:, :, :3]
            
            if args.stain_normalization_method == 'skimage_hist_match':
                gen_rgb_normalized = normalize_staining_rgb_skimage_hist_match(
                    gen_rgb_original, 
                    real_rgb_for_norm
                )
            # Add other methods here if supported by args.stain_normalization_method
            # elif args.stain_normalization_method == 'macenko':
            #     # gen_rgb_normalized = normalize_staining_macenko(...) # Placeholder
            #     logger.warning("Macenko not yet implemented here, using original.")
            #     gen_rgb_normalized = gen_rgb_original
            else:
                logger.warning(f"Unsupported stain normalization method: {args.stain_normalization_method}. Skipping normalization.")
                gen_rgb_normalized = gen_rgb_original # Fallback to original
            
            # Combine normalized RGB with original auxiliary channels
            if num_channels > 3:
                gen_aux_channels = gen_img_np[:, :, 3:]
                gen_img_np = np.concatenate((gen_rgb_normalized, gen_aux_channels), axis=2)
            else:
                gen_img_np = gen_rgb_normalized
        # <<< END STAIN NORMALIZATION MODIFICATION >>>
        
        # Display RGB composites
        axes[0, i].imshow(np.clip(real_img_np[:,:,:3], 0, 1))
        axes[0, i].set_title(f"Real: {current_display_id[:10]}", fontsize=8)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(gen_img_np[:,:,:3], 0, 1))
        axes[1, i].set_title(f"Gen: {current_display_id[:10]}", fontsize=8)
        axes[1, i].axis('off')
        
        # Save RGB representations
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{current_display_id}_real_rgb.png"),
            np.clip(real_img_np[:,:,:3], 0, 1)
        )
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{current_display_id}_gen_rgb.png"),
            np.clip(gen_img_np[:,:,:3], 0, 1)
        )
        
        # Display each extra channel separately
        for c_idx_offset in range(num_extra_channels):
            channel_index = 3 + c_idx_offset
            real_row_idx = 2 + (2 * c_idx_offset)
            gen_row_idx = 3 + (2 * c_idx_offset)
            
            if num_rows > real_row_idx : # Check if row exists
                axes[real_row_idx, i].imshow(np.clip(real_img_np[:,:,channel_index],0,1), cmap='gray')
                axes[real_row_idx, i].set_title(f"Real Ch{channel_index}", fontsize=8)
                axes[real_row_idx, i].axis('off')
            
                plt.imsave(
                    os.path.join(args.output_dir, "generated_images", f"{current_display_id}_real_ch{channel_index}.png"),
                    np.clip(real_img_np[:,:,channel_index],0,1),
                    cmap='gray'
                )

            if num_rows > gen_row_idx: # Check if row exists
                axes[gen_row_idx, i].imshow(np.clip(gen_img_np[:,:,channel_index],0,1), cmap='gray')
                axes[gen_row_idx, i].set_title(f"Gen Ch{channel_index}", fontsize=8)
                axes[gen_row_idx, i].axis('off')

                plt.imsave(
                    os.path.join(args.output_dir, "generated_images", f"{current_display_id}_gen_ch{channel_index}.png"),
                    np.clip(gen_img_np[:,:,channel_index],0,1),
                    cmap='gray'
                )

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "generation_results.png"))
    plt.close(fig) # Close the figure to free memory
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()