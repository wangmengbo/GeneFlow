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

# Ensure project root is in path for src, rectified imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.single_model import RNAtoHnEModel
from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata, analyze_gene_importance
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
# Import the MODIFIED train_with_rectified_flow
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
    parser.add_argument('--relation_rank', type=int, default=50,
                        help='Rank K for low-rank factorization in gene relation network (default: 50).')
    parser.add_argument('--num_aggregation_heads', type=int, default=4,
                        help='Number of heads for cell aggregation in MultiCellRNAEncoder (multi-cell only, default: 4).')

    # New argument for resuming
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint if available.')

    parser = setup_parser(parser) # setup_parser adds args like 'adata', 'concat_mask', 'only_inference' etc.
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Checkpoint and Resume Logic ---
    latest_checkpoint_fname = f"latest_{args.model_type}_checkpoint_rectified.pt"
    latest_checkpoint_path = os.path.join(args.output_dir, latest_checkpoint_fname)
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_rna_to_hne_model_rectified.pt")

    start_epoch = 0
    initial_train_losses, initial_val_losses = [], []
    initial_best_val_loss = float('inf')
    initial_epochs_no_improve = 0
    loaded_optimizer_state_dict = None
    loaded_scheduler_state_dict = None
    loaded_scaler_state_dict = None
    loaded_model_state_dict = None # To store model state if resuming

    # Set initial random seeds. These will be overridden if resuming from a checkpoint.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # import random; random.seed(args.seed) # If using 'random' module

    if args.resume and os.path.exists(latest_checkpoint_path):
        logger.info(f"Attempting to resume training from {latest_checkpoint_path}")
        try:
            # Load checkpoint (weights_only=False is default and correct here)
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            
            loaded_model_state_dict = checkpoint['model_state_dict']
            loaded_optimizer_state_dict = checkpoint['optimizer_state_dict']
            loaded_scheduler_state_dict = checkpoint.get('scheduler_state_dict', None) # .get for backward compatibility
            loaded_scaler_state_dict = checkpoint.get('scaler_state_dict', None)

            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            initial_train_losses = checkpoint.get('train_losses', [])
            initial_val_losses = checkpoint.get('val_losses', [])
            initial_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            initial_epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            
            # Restore RNG states
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu()) # Ensure RNG state is on CPU before loading
            if 'np_rng_state' in checkpoint:
                np.random.set_state(checkpoint['np_rng_state'])
            # if 'random_rng_state' in checkpoint: random.setstate(checkpoint['random_rng_state'])
            
            logger.info(f"Successfully resumed from checkpoint. Training will start at epoch {start_epoch}.")
            logger.info(f"Previous best_val_loss: {initial_best_val_loss:.4f}, epochs_no_improve: {initial_epochs_no_improve}")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Training will start from scratch.")
            # Reset variables to ensure a clean start if checkpoint loading failed
            start_epoch = 0
            initial_train_losses, initial_val_losses = [], []
            initial_best_val_loss = float('inf')
            initial_epochs_no_improve = 0
            loaded_model_state_dict = None
            loaded_optimizer_state_dict = None
            loaded_scheduler_state_dict = None
            loaded_scaler_state_dict = None
            torch.manual_seed(args.seed); np.random.seed(args.seed) # Re-seed
    elif args.resume:
        logger.info(f"Resume flag is set, but checkpoint {latest_checkpoint_path} not found. Starting training from scratch.")
    else:
        logger.info("Starting training from scratch (no resume flag or checkpoint not found).")
    # --- End Checkpoint and Resume Logic ---

    # Load gene expression data (AFTER potential RNG restoration for reproducible splits)
    missing_gene_symbols_list = None # Default if not loaded
    if hasattr(args, 'adata') and args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
    gene_names = expr_df.columns.tolist()

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")
        with open(args.image_paths, "r") as f: image_paths_dict = json.load(f)
        image_paths_dict = {k: v for k, v in image_paths_dict.items() if os.path.exists(v)}
        logger.info(f"Loaded {len(image_paths_dict)} existing cell image paths")
        
        dataset = CellImageGeneDataset(
            expr_df, image_paths_dict, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols_list,
            normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell model
        logger.info("Creating multi-cell dataset")
        with open(args.patch_cell_mapping, "r") as f: patch_to_cells = json.load(f)
        patch_image_paths_dict = None
        if args.patch_image_paths:
            with open(args.patch_image_paths, "r") as f: patch_image_paths_dict = json.load(f)
            patch_image_paths_dict = {k: v for k, v in patch_image_paths_dict.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(patch_image_paths_dict)} existing patch image paths")
            
        dataset = PatchImageGeneDataset(
            expr_df=expr_df, patch_image_paths=patch_image_paths_dict, patch_to_cells=patch_to_cells,
            img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            normalize_aux=args.normalize_aux,
        )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        logger.error(f"Dataset too small for 80/20 split (Total: {len(dataset)}). Adjust data or split ratio.")
        if len(dataset) > 0 and train_size == 0: # Use all for train if val is 0 but train could be non-zero
             train_size = len(dataset)
             val_size = 0
             logger.warning("Using entire dataset for training as validation set would be empty.")
        elif len(dataset) > 0 and val_size == 0: # Should not happen if train_size > 0
            logger.warning("Validation set is empty. Consider a larger dataset.")
        else:
            return # Exit if dataset is completely empty

    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else: # No validation set
        train_dataset = dataset
        val_dataset = torch.utils.data.Subset(dataset, []) # Empty subset for val_loader


    collate_fn_to_use = patch_collate_fn if args.model_type == 'multi' else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=collate_fn_to_use, drop_last=True # drop_last can help with stability for some models
    )
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=collate_fn_to_use
        )
    else: # Create a dummy val_loader if no validation data
        val_loader = None
        logger.warning("Validation set is empty. Validation metrics will not be computed. Early stopping based on validation loss is disabled.")
        initial_best_val_loss = -float('inf') # Ensure any 'improvement' is not triggered
        args.patience = args.epochs + 1 # Effectively disable patience based on val_loss


    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Initialize appropriate model
    gene_dim = expr_df.shape[1]
    # Get concat_mask from args, default to False if not present
    concat_mask_val = args.concat_mask if hasattr(args, 'concat_mask') else False

    model_constructor_args = dict(
        rna_dim=gene_dim, img_channels=args.img_channels, img_size=args.img_size,
        model_channels=128, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1,
        channel_mult=(1, 2, 2, 2), use_checkpoint=False, num_heads=2, num_head_channels=16,
        use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=True,
        concat_mask=concat_mask_val, relation_rank=args.relation_rank,
    )

    if args.model_type == 'single':
        logger.info("Initializing single-cell model")
        # Remove multi-cell specific args if any were passed erroneously for single-cell
        single_model_args = {k: v for k, v in model_constructor_args.items() if k not in ['num_aggregation_heads']}
        model = RNAtoHnEModel(**single_model_args)
    else:  # multi-cell model
        logger.info("Initializing multi-cell model")
        # Pass num_aggregation_heads for multi-cell model
        model = MultiCellRNAtoHnEModel(**model_constructor_args, num_aggregation_heads=args.num_aggregation_heads)
    
    if loaded_model_state_dict:
        model.load_state_dict(loaded_model_state_dict)
        logger.info("Loaded model state from checkpoint into initialized model.")
    model.to(device)
    logger.info(f"Model initialized with gene dimension: {gene_dim}")
    
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")
    
    # Check for only_inference attribute
    only_inference_flag = args.only_inference if hasattr(args, 'only_inference') else False

    if not only_inference_flag:
        if start_epoch >= args.epochs:
            logger.info(f"Training already completed up to epoch {start_epoch -1} (total epochs set to {args.epochs}). Skipping training.")
            # Load losses from the checkpoint if we are skipping training but want to plot
            if os.path.exists(latest_checkpoint_path):
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
            else: # Should not happen if start_epoch > 0
                train_losses, val_losses = [],[]
        else:
            logger.info(f"Calling train_with_rectified_flow. Start epoch: {start_epoch}, Total epochs: {args.epochs}")
            train_losses, val_losses = train_with_rectified_flow(
                model=model, train_loader=train_loader, val_loader=val_loader,
                rectified_flow=rectified_flow, device=device, num_epochs=args.epochs, lr=args.lr,
                best_model_path=best_model_path, latest_checkpoint_path=latest_checkpoint_path,
                patience=args.patience if val_loader is not None else args.epochs + 1, # Disable patience if no val_loader
                use_amp=args.use_amp, weight_decay=args.weight_decay,
                is_multi_cell=(args.model_type == 'multi'),
                start_epoch=start_epoch,
                initial_train_losses=initial_train_losses, initial_val_losses=initial_val_losses,
                initial_best_val_loss=initial_best_val_loss if val_loader is not None else -float('inf'),
                initial_epochs_no_improve=initial_epochs_no_improve,
                optimizer_state_dict=loaded_optimizer_state_dict,
                scheduler_state_dict=loaded_scheduler_state_dict,
                scaler_state_dict=loaded_scaler_state_dict
            )
        
        logger.info(f"Training process finished. Best model should be at {best_model_path}")
        
        if train_losses or val_losses: # Plot if there's any loss data
            plt.figure(figsize=(10, 5))
            if train_losses: plt.plot(train_losses, label='Train Loss')
            if val_losses: plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
            logger.info("Saved training curves.")
        else:
            logger.info("No training/validation losses to plot (possibly skipped training or training failed early).")
    else:
        logger.info(f"Skipping training due to --only_inference. Using existing model at {best_model_path}")

    # Load best model for evaluation/generation
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found at {best_model_path}. Cannot proceed with generation/analysis.")
        if hasattr(args, 'only_importance_anlaysis') and args.only_importance_anlaysis:
             return # Exit if only importance analysis and no model
        logger.warning("If training was just completed, the best model should exist. Check training logs.")
        return

    logger.info(f"Loading best model from {best_model_path} for post-training tasks.")
    # Using weights_only=False is safer if the checkpoint might contain more than just weights
    # The original script used weights_only=True, assuming "model" key and simple other types.
    # The modified train function saves epoch and val_loss which are simple, so True might still work.
    # However, for robustness, especially if other non-tensor data might be added, False is better.
    try:
        checkpoint = torch.load(best_model_path, map_location=device) # weights_only=False
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else: # If the checkpoint is just the state_dict
            model.load_state_dict(checkpoint)
        logger.info("Successfully loaded best model state_dict.")
    except Exception as e:
        logger.error(f"Failed to load best model from {best_model_path}: {e}")
        logger.info("Attempting to load with weights_only=True (original behavior).")
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            logger.info("Successfully loaded best model state_dict using weights_only=True.")
        except Exception as e2:
            logger.error(f"Fallback loading with weights_only=True also failed: {e2}. Cannot proceed.")
            return

    model.to(device)
    model.eval()
    
    # --- Image Generation and Gene Importance Analysis ---
    if hasattr(args, 'only_importance_anlaysis') and args.only_importance_anlaysis:
        logger.info("Skipping image generation as only importance analysis is requested.")
    elif val_size == 0: # Cannot generate from validation set if it's empty
        logger.warning("Skipping image generation from validation set as it is empty.")
    else:
        logger.info("Generating images from validation set...")
        num_vis_samples = min(10, len(val_dataset)) # val_dataset is guaranteed to exist if val_size > 0
        vis_indices = torch.randperm(len(val_dataset))[:num_vis_samples]
        vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)

        vis_loader_collate_fn = patch_collate_fn if args.model_type == 'multi' else None
        vis_loader = DataLoader(
            vis_dataset, batch_size=num_vis_samples, shuffle=False, collate_fn=vis_loader_collate_fn
        )
        batch = next(iter(vis_loader))
        
        gen_params_for_func = {
            "model": model, "rectified_flow": rectified_flow, "device": device,
            "num_steps": args.gen_steps, "is_multi_cell": (args.model_type == 'multi')
        }
        if args.model_type == 'single':
            gene_expr_gen = batch['gene_expr'].to(device)
            real_images = batch['image']
            display_ids = batch['cell_id']
            gene_mask_gen = batch.get('gene_mask', None)
            if gene_mask_gen is not None: gen_params_for_func["gene_mask"] = gene_mask_gen.to(device)
            gen_params_for_func["gene_expr"] = gene_expr_gen
        else:  # multi-cell model
            processed_batch = prepare_multicell_batch(batch, device)
            gen_params_for_func["gene_expr"] = processed_batch['gene_expr']
            gen_params_for_func["num_cells"] = processed_batch['num_cells'] # num_cells from collate_fn
            real_images = batch['image']
            display_ids = batch['patch_id']
        
        generated_images = generate_images_with_rectified_flow(**gen_params_for_func)

        gen_img_dir = os.path.join(args.output_dir, "generated_images")
        os.makedirs(gen_img_dir, exist_ok=True)
        num_img_channels = args.img_channels
        num_extra_channels = max(0, num_img_channels - 3)
        num_rows_plot = 2 + (2 * num_extra_channels)

        fig, axes = plt.subplots(num_rows_plot, num_vis_samples, figsize=(3*num_vis_samples, 2*num_rows_plot + 1))
        if num_rows_plot == 1 and num_vis_samples == 1: axes = np.array([[axes]])
        elif num_rows_plot == 1: axes = np.expand_dims(axes, axis=0)
        elif num_vis_samples == 1: axes = np.expand_dims(axes, axis=1)

        for i in range(num_vis_samples):
            real_img_np = real_images[i].cpu().numpy().transpose(1, 2, 0)
            gen_img_np = generated_images[i].cpu().numpy().transpose(1, 2, 0)
            current_id = display_ids[i] if i < len(display_ids) else f"sample_{i}"

            axes[0, i].imshow(np.clip(real_img_np[:,:,:3],0,1))
            axes[0, i].set_title(f"Real RGB: {str(current_id)[:10]}")
            axes[0, i].axis('off')
            plt.imsave(os.path.join(gen_img_dir, f"{current_id}_real_rgb.png"), np.clip(real_img_np[:,:,:3],0,1))
            
            axes[1, i].imshow(np.clip(gen_img_np[:,:,:3],0,1))
            axes[1, i].set_title(f"Gen RGB: {str(current_id)[:10]}")
            axes[1, i].axis('off')
            plt.imsave(os.path.join(gen_img_dir, f"{current_id}_gen_rgb.png"), np.clip(gen_img_np[:,:,:3],0,1))
            
            for c_idx in range(num_extra_channels):
                channel_val = 3 + c_idx
                real_row_idx, gen_row_idx = 2 + (2*c_idx), 3 + (2*c_idx)
                axes[real_row_idx, i].imshow(real_img_np[:,:,channel_val], cmap='gray'); axes[real_row_idx, i].set_title(f"Real Ch{channel_val}"); axes[real_row_idx, i].axis('off')
                plt.imsave(os.path.join(gen_img_dir, f"{current_id}_real_ch{channel_val}.png"), real_img_np[:,:,channel_val], cmap='gray')
                axes[gen_row_idx, i].imshow(gen_img_np[:,:,channel_val], cmap='gray'); axes[gen_row_idx, i].set_title(f"Gen Ch{channel_val}"); axes[gen_row_idx, i].axis('off')
                plt.imsave(os.path.join(gen_img_dir, f"{current_id}_gen_ch{channel_val}.png"), gen_img_np[:,:,channel_val], cmap='gray')
        plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, "generation_results.png"))
        logger.info(f"Generated images and comparison plot saved to {args.output_dir}")

    # Gene importance analysis
    if args.model_type == 'single' and val_loader is not None:
        logger.info("Performing gene importance analysis for single-cell model.")
        importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
        analysis_timesteps_val = args.analysis_timesteps if hasattr(args, 'analysis_timesteps') else [0.1, 0.5, 0.9]
        analysis_batches_val = args.analysis_batches if hasattr(args, 'analysis_batches') else 10
        
        analyze_gene_importance(
            model=model, data_loader=val_loader, rectified_flow=rectified_flow, device=device,
            gene_names=gene_names, output_path=importance_output_path,
            timesteps_to_analyze=analysis_timesteps_val,
            num_batches_to_analyze=analysis_batches_val
        )
        logger.info(f"Gene importance scores saved to {importance_output_path}")
    elif args.model_type == 'multi' and (hasattr(args, 'only_importance_anlaysis') and args.only_importance_anlaysis):
         logger.warning("Gene importance analysis is currently supported only for single-cell models or not run if val_loader is missing.")
    elif val_loader is None and (hasattr(args, 'only_importance_anlaysis') and args.only_importance_anlaysis):
        logger.warning("Cannot run gene importance analysis because the validation loader is not available (val_size was 0).")


    logger.info(f"All tasks completed. Outputs are in {args.output_dir}")

if __name__ == "__main__":
    main()