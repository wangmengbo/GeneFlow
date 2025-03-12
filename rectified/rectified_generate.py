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
from src.dataset import (CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn,
                        load_preprocessed_hest1k_singlecell_data,
                        OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn,
                        FastSeparatePatchDataset, fast_separate_patch_collate_fn)
# from rectified.rectified_train_ddp import generate_images_with_rectified_flow
from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match
from rectified.rectified_utils import generate_images_with_rectified_flow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate images using pretrained RNA to H&E model.")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate.')
    
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
    expr_df = None
    gene_names = None
    missing_gene_symbols_list = None

    # Data loading logic
    if args.hest1k_base_dir:
        if args.hest1k_sid is None or len(args.hest1k_sid) == 0:
            hest_metadata = pd.read_csv("/depot/natallah/data/Mengbo/HnE_RNA/data/HEST-1k/data/HEST_v1_1_0.csv")
            args.hest1k_sid = hest_metadata[(hest_metadata['st_technology']=='Xenium') & 
                                           (hest_metadata['species']=='Homo sapiens')]['id'].tolist()
        logger.info(f"Loading pre-processed HEST-1k data for sample {args.hest1k_sid}")
        expr_df, image_paths_dict = load_preprocessed_hest1k_singlecell_data(
            args.hest1k_sid, args.hest1k_base_dir, img_size=args.img_size, img_channels=args.img_channels)
        missing_gene_symbols_list = None
    elif args.hest1k_xenium_dir or args.hest1k_xenium_fast_dir:
        logger.info(f"Preparing to load manually processed HEST-1k Xenium samples")
        # expr_df will remain None for multi-sample datasets
    elif args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
        missing_gene_symbols_list = None

    if expr_df is not None:
        logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
        gene_names = expr_df.columns.tolist()

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")
        if not (args.hest1k_sid and args.hest1k_base_dir):
            # Load image paths if provided (for visualization)
            if args.image_paths:
                logger.info(f"Loading image paths from {args.image_paths}")
                with open(args.image_paths, "r") as f:
                    image_paths_dict = json.load(f)
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
            missing_gene_symbols=missing_gene_symbols_list,
            normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell model
        logger.info("Creating multi-cell dataset")
        
        # Check for fast reformatted data first
        if args.hest1k_xenium_fast_dir is not None:
            logger.info(f"Loading fast reformatted HEST-1k Xenium dataset from {args.hest1k_xenium_fast_dir}")
            dataset = FastSeparatePatchDataset(
                reformatted_dir=args.hest1k_xenium_fast_dir,
                sample_metadata_csv=args.hest1k_xenium_metadata,
                sample_ids=args.hest1k_xenium_samples,
                img_size=args.img_size,
                img_channels=args.img_channels,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size), antialias=True),
                ]),
                filter_unassigned=True,
                min_gene_samples=1,
                cache_unified_genes=True
            )
            gene_names = dataset.gene_names
            logger.info(f"Fast dataset loaded: {len(dataset)} patches, {len(gene_names)} unified genes")
            
        elif args.hest1k_xenium_dir is not None:
            logger.info(f"Loading HEST-1k Xenium on-demand dataset from {args.hest1k_xenium_dir}")
            dataset = OnDemandMultiSampleHestXeniumDataset(
                combined_dir=args.hest1k_xenium_dir,
                sample_metadata_csv=args.hest1k_xenium_metadata,
                sample_ids=args.hest1k_xenium_samples,
                img_size=args.img_size,
                img_channels=args.img_channels,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size), antialias=True),
                ]),
                normalize_aux=args.normalize_aux,
                cache_metadata=True
            )
            # Print sample statistics
            sample_stats = dataset.get_sample_stats()
            logger.info("Sample statistics:")
            for sample_id, stats in sample_stats.items():
                logger.info(f" {sample_id}: {stats['n_patches']} patches, "
                           f"{stats['n_cells']} cells, {stats['n_genes']} genes")
        else:
            # Load patch-to-cell mapping
            logger.info(f"Loading patch-to-cell mapping from {args.patch_cell_mapping}")
            with open(args.patch_cell_mapping, "r") as f:
                patch_to_cells = json.load(f)

            # Load patch image paths if provided
            if args.patch_image_paths:
                logger.info(f"Loading patch image paths from {args.patch_image_paths}")
                with open(args.patch_image_paths, "r") as f:
                    patch_image_paths_dict = json.load(f)
                
                # Filter out non-existent files
                patch_image_paths_tmp = {}
                for k,v in patch_image_paths_dict.items():
                    if os.path.exists(v):
                        patch_image_paths_tmp[k] = v
                patch_image_paths_dict = patch_image_paths_tmp
                logger.info(f"After filtering: {len(patch_image_paths_dict)} valid patch image paths")
            else:
                patch_image_paths_dict = None

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
    if num_vis_samples == 0 and len(dataset) > 0:
        num_vis_samples = 1
    elif len(dataset) == 0:
        logger.error("Dataset is empty after splitting. Exiting.")
        return

    vis_indices = torch.randperm(len(dataset))[:num_vis_samples].tolist()
    vis_dataset = torch.utils.data.Subset(dataset, vis_indices)

    # Use the appropriate collate function based on model type and dataset type
    if args.model_type == 'multi':
        # Determine collate function based on dataset type
        if args.hest1k_xenium_fast_dir is not None:
            collate_fn = fast_separate_patch_collate_fn
            logger.info("Using fast separate patch collate function")
        elif args.hest1k_xenium_dir is not None:
            collate_fn = multi_sample_hest_xenium_collate_fn
            logger.info("Using multi-sample Xenium collate function")
        else:
            collate_fn = patch_collate_fn
            logger.info("Using standard patch collate function")
            
        vis_loader = DataLoader(
            vis_dataset,
            batch_size=args.batch_size if num_vis_samples > args.batch_size else num_vis_samples,
            shuffle=False,
            num_workers=args.num_dataloader_workers,
            collate_fn=collate_fn
        )
    else:
        vis_loader = DataLoader(
            vis_dataset,
            batch_size=args.batch_size if num_vis_samples > args.batch_size else num_vis_samples,
            shuffle=False,
            num_workers=args.num_dataloader_workers
        )

    # Initialize appropriate model based on model type
    if expr_df is not None:
        gene_dim = expr_df.shape[1]
    elif hasattr(dataset, 'gene_names'):
        # For multi-sample datasets, get gene dimension from dataset
        gene_dim = len(dataset.gene_names)
        gene_names = dataset.gene_names
    else:
        gene_dim = len(gene_names)

    if gene_dim is None:
        raise ValueError("Could not determine gene dimension. Ensure dataset has gene_names attribute.")
    
    logger.info(f"Gene dimension: {gene_dim}")

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
        concat_mask=getattr(args, 'concat_mask', False),
        relation_rank=getattr(args, 'relation_rank', 50),
        use_multi_head_attention=getattr(args, 'use_multi_head_attention', True),
        use_feature_gating=getattr(args, 'use_feature_gating', True),
        use_residual_blocks=getattr(args, 'use_residual_blocks', True),
        use_layer_norm=getattr(args, 'use_layer_norm', True),
        use_gene_relations=getattr(args, 'use_gene_relations', True),
    )

    # Load the pretrained model
    logger.info(f"Loading pretrained model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        raise KeyError(f"Checkpoint missing model weights. Available keys: {checkpoint.keys()}")
    
    model_config_ckpt = checkpoint.get("config", {})

    # Update args from model_config_ckpt if they were used for training
    args.img_channels = model_config_ckpt.get('img_channels', args.img_channels)
    args.img_size = model_config_ckpt.get('img_size', args.img_size)

    current_model_type = model_config_ckpt.get('model_type', args.model_type)

    model = None
    try:
        if current_model_type == 'single':
            logger.info("Initializing single-cell model")
            model_constructor_args['use_gene_attention'] = getattr(args, 'use_gene_attention', True)
            model = RNAtoHnEModel(**model_constructor_args)
        else:
            logger.info("Initializing multi-cell model")
            model_constructor_args['num_aggregation_heads'] = getattr(args, 'num_aggregation_heads', 4)
            model = MultiCellRNAtoHnEModel(**model_constructor_args)
        
        # Handle module. prefix for DDP checkpoints
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        model.load_state_dict(model_state)
    except Exception as e:
        logger.warning(f"Failed to load model with current constructor: {e}. Trying deprecated constructor.")
        deprecated_constructor_args = model_constructor_args.copy()
        if 'relation_rank' in deprecated_constructor_args and current_model_type == 'single':
            deprecated_constructor_args.pop('relation_rank')
        if 'num_aggregation_heads' in deprecated_constructor_args:
            deprecated_constructor_args.pop('num_aggregation_heads')
        
        if current_model_type == 'single':
            model = RNAtoHnEModel_deprecation(**deprecated_constructor_args)
        else:
            model = MultiCellRNAtoHnEModel_deprecation(**deprecated_constructor_args)
        model.load_state_dict(model_state)

    logger.info(f"Model loaded successfully using {current_model_type} constructor.")
    model.to(device)
    model.eval()

    # Initialize the rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")

    # Get a batch of data
    # Process all batches
    all_generated_images = []
    all_real_images = []
    all_display_ids = []
    
    logger.info(f"Processing {len(vis_loader)} batches...")
    
    for batch_idx, batch in enumerate(vis_loader):
        logger.info(f"Generating batch {batch_idx+1}/{len(vis_loader)}...")
        
        # Handle data differently based on model type
        if current_model_type == 'single':
            gene_expr = batch['gene_expr'].to(device)
            real_images_tensor = batch['image']
            display_ids = batch['cell_id']
            gene_mask = batch.get('gene_mask', None)
            if gene_mask is not None:
                gene_mask = gene_mask.to(device)

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
            num_cells_info = processed_batch['num_cells']
            real_images_tensor = batch['image']
            display_ids = batch['patch_id']

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
        
        # Accumulate results
        all_generated_images.append(generated_images_tensor.cpu())
        all_real_images.append(real_images_tensor.cpu())
        all_display_ids.extend(display_ids)
    
    logger.info("Image generation complete")
    
    # Concatenate all batches
    generated_images_tensor = torch.cat(all_generated_images, dim=0)
    real_images_tensor = torch.cat(all_real_images, dim=0)
    display_ids = all_display_ids
    
    logger.info(f"Generated {len(generated_images_tensor)} images total")
    
    num_channels = args.img_channels  # Define num_channels for saving loop

    # Save individual images
    for i in range(len(generated_images_tensor)):
        real_img_np = real_images_tensor[i].numpy().transpose(1, 2, 0)
        gen_img_np = generated_images_tensor[i].numpy().transpose(1, 2, 0)
        current_display_id = display_ids[i] if i < len(display_ids) else f"Sample_{i}"

        # Stain normalization if enabled
        if args.enable_stain_normalization and num_channels >= 3:
            logger.info(f"Applying stain normalization to sample {current_display_id}")
            real_rgb_for_norm = real_img_np[:, :, :3]
            gen_rgb_original = gen_img_np[:, :, :3]

            if args.stain_normalization_method == 'skimage_hist_match':
                gen_rgb_normalized = normalize_staining_rgb_skimage_hist_match(
                    gen_rgb_original,
                    real_rgb_for_norm
                )
            else:
                gen_rgb_normalized = gen_rgb_original

            if num_channels > 3:
                gen_aux_channels = gen_img_np[:, :, 3:]
                gen_img_np = np.concatenate((gen_rgb_normalized, gen_aux_channels), axis=2)
            else:
                gen_img_np = gen_rgb_normalized

        # Save RGB images
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{current_display_id}_real_rgb.png"),
            np.clip(real_img_np[:,:,:3], 0, 1)
        )
        plt.imsave(
            os.path.join(args.output_dir, "generated_images", f"{current_display_id}_gen_rgb.png"),
            np.clip(gen_img_np[:,:,:3], 0, 1)
        )

        # Save extra channels if they exist
        for c_idx in range(3, num_channels):
            plt.imsave(
                os.path.join(args.output_dir, "generated_images", f"{current_display_id}_real_ch{c_idx}.png"),
                np.clip(real_img_np[:,:,c_idx], 0, 1),
                cmap='gray'
            )
            plt.imsave(
                os.path.join(args.output_dir, "generated_images", f"{current_display_id}_gen_ch{c_idx}.png"),
                np.clip(gen_img_np[:,:,c_idx], 0, 1),
                cmap='gray'
            )
    
    # Create multi-page PDF with all samples (2 pairs per row: real1, gen1, real2, gen2)
    from matplotlib.backends.backend_pdf import PdfPages
    
    num_samples = len(generated_images_tensor)
    pairs_per_row = 2  # Number of sample pairs per row
    rows_per_page = 10  # Number of rows per page
    samples_per_page = pairs_per_row * rows_per_page  # 2 samples per row * 10 rows = 20 samples per page
    num_pages = (num_samples + samples_per_page - 1) // samples_per_page
    
    logger.info(f"Creating {num_pages}-page PDF with all {num_samples} samples (2 pairs per row)...")
    
    pdf_path = os.path.join(args.output_dir, "generation_results.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for page_idx in range(num_pages):
            start_idx = page_idx * samples_per_page
            end_idx = min(start_idx + samples_per_page, num_samples)
            num_samples_this_page = end_idx - start_idx
            num_rows_this_page = (num_samples_this_page + pairs_per_row - 1) // pairs_per_row
            
            # Create figure: 4 columns (real1, gen1, real2, gen2)
            fig_height = 3 * num_rows_this_page
            fig, axes = plt.subplots(num_rows_this_page, 4, figsize=(16, fig_height))
            
            # Handle single row case
            if num_rows_this_page == 1:
                axes = axes.reshape(1, -1)
            
            for row_idx in range(num_rows_this_page):
                for pair_idx in range(pairs_per_row):
                    sample_idx = start_idx + row_idx * pairs_per_row + pair_idx
                    
                    if sample_idx >= end_idx:
                        # Hide unused subplots in last row if odd number of samples
                        axes[row_idx, pair_idx * 2].axis('off')
                        axes[row_idx, pair_idx * 2 + 1].axis('off')
                        continue
                    
                    real_img_np = real_images_tensor[sample_idx].numpy().transpose(1, 2, 0)
                    gen_img_np = generated_images_tensor[sample_idx].numpy().transpose(1, 2, 0)
                    current_display_id = display_ids[sample_idx]
                    
                    col_real = pair_idx * 2
                    col_gen = pair_idx * 2 + 1
                    
                    # Real image
                    axes[row_idx, col_real].imshow(np.clip(real_img_np[:,:,:3], 0, 1))
                    axes[row_idx, col_real].set_title(f"Real: {current_display_id}", fontsize=9)
                    axes[row_idx, col_real].axis('off')
                    
                    # Generated image
                    axes[row_idx, col_gen].imshow(np.clip(gen_img_np[:,:,:3], 0, 1))
                    axes[row_idx, col_gen].set_title(f"Gen: {current_display_id}", fontsize=9)
                    axes[row_idx, col_gen].axis('off')
            
            # Add page number
            fig.suptitle(f"Page {page_idx + 1}/{num_pages} (Samples {start_idx + 1}-{end_idx})", 
                        fontsize=14, y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            
            logger.info(f"Completed page {page_idx + 1}/{num_pages}")
    
    logger.info(f"PDF saved to {pdf_path}")
    
    # Also create a PNG with first 10 samples for quick preview (2 pairs per row)
    logger.info("Creating PNG preview with first 10 samples...")
    num_preview = min(10, num_samples)
    num_preview_rows = (num_preview + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(num_preview_rows, 4, figsize=(16, 3 * num_preview_rows))
    if num_preview_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx in range(num_preview_rows):
        for pair_idx in range(2):
            sample_idx = row_idx * 2 + pair_idx
            
            if sample_idx >= num_preview:
                axes[row_idx, pair_idx * 2].axis('off')
                axes[row_idx, pair_idx * 2 + 1].axis('off')
                continue
            
            real_img_np = real_images_tensor[sample_idx].numpy().transpose(1, 2, 0)
            gen_img_np = generated_images_tensor[sample_idx].numpy().transpose(1, 2, 0)
            current_display_id = display_ids[sample_idx]
            
            col_real = pair_idx * 2
            col_gen = pair_idx * 2 + 1
            
            axes[row_idx, col_real].imshow(np.clip(real_img_np[:,:,:3], 0, 1))
            axes[row_idx, col_real].set_title(f"Real: {current_display_id}", fontsize=9)
            axes[row_idx, col_real].axis('off')
            
            axes[row_idx, col_gen].imshow(np.clip(gen_img_np[:,:,:3], 0, 1))
            axes[row_idx, col_gen].set_title(f"Gen: {current_display_id}", fontsize=9)
            axes[row_idx, col_gen].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "generation_results.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"- PDF with all {num_samples} samples: generation_results.pdf ({samples_per_page} samples per page)")
    logger.info(f"- PNG preview with first {num_preview} samples: generation_results.png")
    logger.info(f"- Individual images: generated_images/")


if __name__ == "__main__":
    main()
