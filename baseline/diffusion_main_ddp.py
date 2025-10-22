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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.single_model import RNAtoHnEModel
from baseline.diffusion import GaussianDiffusion
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from baseline.diffusion_train_ddp import train_with_diffusion, generate_images_with_diffusion
from src.utils import setup_parser, parse_adata, analyze_gene_importance_diffusion
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn, \
    load_preprocessed_hest1k_singlecell_data, \
        OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn, \
            FastSeparatePatchDataset, fast_separate_patch_collate_fn

# Configure logging
def setup_logging(rank):
    """Setup logging for DDP - only rank 0 logs to avoid spam"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)

# ============================================================
# DDP Setup and Cleanup Functions
# ============================================================

def setup_ddp():
    """Initialize DDP environment"""
    # Get rank and world size from environment (set by torchrun)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, local_rank, world_size, device

def cleanup_ddp():
    """Cleanup DDP environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================
# Checkpoint Save and Load Functions (DDP-aware)
# ============================================================

def save_checkpoint(state, filename, rank):
    """Save checkpoint only on rank 0"""
    if rank == 0:
        torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, device):
    """Load checkpoint and handle DDP model state dict"""
    checkpoint = torch.load(filename, map_location=device)
    
    # Handle DDP model state dict (remove 'module.' prefix if present)
    model_state_dict = checkpoint['model_state_dict']
    if isinstance(model, DDP):
        # If loading into DDP model, ensure state dict matches
        if not any(key.startswith('module.') for key in model_state_dict.keys()):
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
    else:
        # If loading into non-DDP model, remove 'module.' prefix
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Ensure optimizer state is on correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return start_epoch, best_val_loss

# ======================================
# Main Function
# ======================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RNA to H&E cell image generator with Diffusion.")
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_diffusion', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels (3 for RGB, 1 Greyscale).')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for training.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--gen_steps', type=int, default=300, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi',help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels.')
    parser.add_argument('--diffusion_timesteps', type=int, default=300, help='Number of timesteps for diffusion process')
    parser.add_argument('--beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Noise schedule for diffusion')
    parser.add_argument('--predict_noise', action='store_true', default=True, help='Whether model predicts noise (True) or x_0 (False)')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method for diffusion generation')
    parser.add_argument('--hest1k_sid', type=str, nargs='*', default=None, help='HEST-1k sample ID for direct loading')
    parser.add_argument('--hest1k_base_dir', type=str, default=None, help='Base directory for HEST-1k data')
    parser.add_argument('--hest1k_xenium_dir', type=str, default=None, help='Directory for HEST-1k Xenium AnnData files') 
    parser.add_argument('--hest1k_xenium_metadata', type=str, default=None, help='Metadata CSV for HEST-1k Xenium data')
    parser.add_argument('--hest1k_xenium_samples', type=str, nargs='*', default=None, help='Specific Xenium sample IDs to use')
    parser.add_argument('--hest1k_xenium_fast_dir', type=str, default=None, 
                   help='Directory for reformatted fast-loading HEST-1k Xenium patch data')
    parser.add_argument('--num_dataloader_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel training.')
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Initialize DDP if requested
    if args.use_ddp:
        rank, local_rank, world_size, device = setup_ddp()
        # Adjust batch size for DDP (total effective batch size = batch_size * world_size)
        original_batch_size = args.batch_size
        args.batch_size = args.batch_size // world_size
        if args.batch_size == 0:
            args.batch_size = 1
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_batch_size = args.batch_size

    # Setup logging
    logger = setup_logging(rank)
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Using device: {device}, Rank: {rank}, World size: {world_size}")
        if args.use_ddp:
            logger.info(f"Per-GPU batch size: {args.batch_size}, Effective batch size: {original_batch_size}")
        else:
            logger.info(f"Batch size: {args.batch_size}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank)  # Different seed per rank for data shuffling
    np.random.seed(args.seed + rank)

    expr_df = None
    gene_names = None
    
    # Data loading logic
    if args.hest1k_base_dir:
        if args.hest1k_sid is None or len(args.hest1k_sid) == 0:
            hest_metadata = pd.read_csv("/depot/natallah/data/Mengbo/HnE_RNA/data/HEST-1k/data/HEST_v1_1_0.csv")
            args.hest1k_sid = hest_metadata[(hest_metadata['st_technology']=='Xenium') & \
                                            (hest_metadata['species']=='Homo sapiens')]['id'].tolist()
        if rank == 0:
            logger.info(f"Loading pre-processed HEST-1k data for sample {args.hest1k_sid}")
        expr_df, image_paths = load_preprocessed_hest1k_singlecell_data(args.hest1k_sid, args.hest1k_base_dir, img_size=args.img_size, img_channels=args.img_channels)
        missing_gene_symbols = None
    elif args.hest1k_xenium_dir or args.hest1k_xenium_fast_dir:
        if rank == 0:
            logger.info(f"Preparing to load manually processed HEST-1k Xenium samples")
    elif args.adata is not None:
        if rank == 0:
            logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        if rank == 0:
            logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
        missing_gene_symbols = None

    if expr_df is not None and rank == 0:
        logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
        gene_names = expr_df.columns.tolist()

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        if rank == 0:
            logger.info("Creating single-cell dataset")
        if not (args.hest1k_sid and args.hest1k_base_dir):
            if rank == 0:
                logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f:
                image_paths = json.load(f)
            if rank == 0:
                logger.info(f"Loaded {len(image_paths)} cell image paths")
            image_paths_tmp = {}
            for k, v in image_paths.items():
                if os.path.exists(v):
                    image_paths_tmp[k] = v
            image_paths = image_paths_tmp
            if rank == 0:
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
        if rank == 0:
            logger.info("Creating multi-cell dataset")
        # Check for fast reformatted data first (recommended)
        if args.hest1k_xenium_fast_dir is not None:
            if rank == 0:
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
            
            if rank == 0:
                logger.info(f"Fast dataset loaded: {len(dataset)} patches, {len(gene_names)} unified genes")
        elif args.hest1k_xenium_dir is not None:
            if rank == 0:
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
            
            if rank == 0:
                sample_stats = dataset.get_sample_stats()
                logger.info("Sample statistics:")
                for sample_id, stats in sample_stats.items():
                    logger.info(f"  {sample_id}: {stats['n_patches']} patches, "
                               f"{stats['n_cells']} cells, {stats['n_genes']} genes")
        else:
            # Load patch-to-cell mapping
            if rank == 0:
                logger.info(f"Loading patch-to-cell mapping from {args.patch_cell_mapping}")
            with open(args.patch_cell_mapping, "r") as f:
                patch_to_cells = json.load(f)
            
            # Load patch image paths if provided
            if args.patch_image_paths:
                if rank == 0:
                    logger.info(f"Loading patch image paths from {args.patch_image_paths}")
                with open(args.patch_image_paths, "r") as f:
                    patch_image_paths = json.load(f)

                patch_image_paths_tmp = {}
                for k,v in patch_image_paths.items():
                    if os.path.exists(v):
                        patch_image_paths_tmp[k] = v
                patch_image_paths = patch_image_paths_tmp
                if rank == 0:
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
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)  # Consistent split across ranks
    )

    # Create distributed samplers
    if args.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            seed=args.seed
        )
        shuffle_train = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    # Data loaders with distributed samplers
    if args.model_type == 'single':
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=args.num_dataloader_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_dataloader_workers,
            pin_memory=True
        )
    else:
        # Determine collate function based on dataset type
        if args.hest1k_xenium_fast_dir is not None:
            collate_fn = fast_separate_patch_collate_fn
            if rank == 0:
                logger.info("Using fast separate patch collate function")
        elif args.hest1k_xenium_dir is not None:
            collate_fn = multi_sample_hest_xenium_collate_fn
            if rank == 0:
                logger.info("Using multi-sample Xenium collate function")
        else:
            collate_fn = patch_collate_fn
            if rank == 0:
                logger.info("Using standard patch collate function")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=args.num_dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )

    if rank == 0:
        logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Initialize appropriate model
    if expr_df is not None:
        gene_dim = expr_df.shape[1]
    elif hasattr(dataset, 'gene_names'):
        gene_dim = len(dataset.gene_names)
        gene_names = dataset.gene_names
    else:
        gene_dim = len(gene_names)

    if gene_dim is None:
        raise ValueError("Could not determine gene dimension. Ensure dataset has gene_names attribute.")
    
    if rank == 0:
        logger.info(f"Gene dimension: {gene_dim}")

    if args.model_type == 'single':
        if rank == 0:
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
        if rank == 0:
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

    # Move model to device
    model.to(device)

    # Wrap model with DDP if using distributed training
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")
    
    if rank == 0:
        logger.info(f"Model initialized with gene dimension: {gene_dim}")

    # Initialize the diffusion model directly on the correct device
    diffusion = GaussianDiffusion(
        timesteps=args.diffusion_timesteps,
        beta_schedule=args.beta_schedule,
        predict_noise=args.predict_noise,
        device=device  # Specify the device here
    )
    if rank == 0:
        logger.info("Initialized diffusion model on device: " + str(device))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoint path
    checkpoint_path = os.path.join(args.output_dir, "training_checkpoint.pt")
    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoint if it exists (only on rank 0, then broadcast)
    if rank == 0 and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Broadcast checkpoint info to all ranks
    if args.use_ddp:
        checkpoint_info = torch.tensor([start_epoch, best_val_loss], device=device)
        dist.broadcast(checkpoint_info, src=0)
        start_epoch, best_val_loss = int(checkpoint_info[0].item()), float(checkpoint_info[1].item())

    # Train model
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_rna_to_hne_model_diffusion.pt")
    if not args.only_inference:
        train_losses, val_losses = train_with_diffusion(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            diffusion=diffusion,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=best_model_path,
            patience=args.patience,
            use_amp=args.use_amp,
            weight_decay=args.weight_decay,
            is_multi_cell=(args.model_type == 'multi'),
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            save_checkpoint_fn=save_checkpoint,
            rank=rank,
            world_size=world_size,
            use_ddp=args.use_ddp
        )

        if rank == 0:
            logger.info(f"Training complete. Best model saved at {best_model_path}")
            
            # Plot training curves
            plt.figure(figsize=(12, 5))
            
            # Plot loss curves
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.8)
            plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot learning rate curve
            plt.subplot(1, 2, 2)
            epochs_range = range(1, len(train_losses) + 1)
            # Calculate LR values (approximate for visualization)
            initial_lr = args.lr
            final_lr = args.lr * 0.01
            lr_values = [initial_lr * (final_lr/initial_lr)**(ep/len(train_losses)) for ep in range(len(train_losses))]
            plt.plot(epochs_range, lr_values, label='Learning Rate', color='green', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
            
            # Save loss values to CSV for further analysis
            loss_df = pd.DataFrame({
                'epoch': range(1, len(train_losses) + 1),
                'train_loss': train_losses,
                'val_loss': val_losses
            })
            loss_df.to_csv(os.path.join(args.output_dir, "training_losses.csv"), index=False)
            logger.info(f"Training curves and loss data saved to {args.output_dir}")
    else:
        if rank == 0:
            logger.info(f"Skipping training. Using existing model at {best_model_path}")

    # Synchronize before inference
    if args.use_ddp:
        dist.barrier()

    # Load best model for evaluation (only on rank 0 for generation)
    if rank == 0:
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=True)
        
        # Handle DDP state dict
        model_state_dict = checkpoint["model"]
        if isinstance(model, DDP):
            if not any(key.startswith('module.') for key in model_state_dict.keys()):
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        else:
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        model.load_state_dict(model_state_dict)

        # Generate images from validation set (only on rank 0)
        logger.info("Generating images from validation set...")
        
        # Skip importance analysis for multi-cell model for now
        if args.model_type == 'single':
            importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
            analyze_gene_importance_diffusion(
                model=model,
                data_loader=val_loader, # Use validation loader
                diffusion=diffusion,
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

        # Create a subset of the validation set for visualization
        num_vis_samples = min(10, len(val_dataset))
        vis_indices = torch.randperm(len(val_dataset))[:num_vis_samples]
        vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)

        # Use the appropriate collate function based on model type
        if args.model_type == 'multi':
            if args.hest1k_xenium_fast_dir is not None:
                vis_collate_fn = fast_separate_patch_collate_fn
            elif args.hest1k_xenium_dir is not None:
                vis_collate_fn = multi_sample_hest_xenium_collate_fn
            else:
                vis_collate_fn = patch_collate_fn
                
            vis_loader = DataLoader(
                vis_dataset, 
                batch_size=num_vis_samples, 
                shuffle=False,
                collate_fn=vis_collate_fn  # Use the same custom collate function
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
            
            # Generate images with diffusion
            generated_images = generate_images_with_diffusion(
                model=model,
                diffusion=diffusion,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                gene_mask=gene_mask,
                is_multi_cell=False,
                method="ddim"  # Use DDIM for faster sampling
            )
        else:  # multi-cell model
            # Prepare batch for multi-cell model
            processed_batch = prepare_multicell_batch(batch, device)
            gene_expr = processed_batch['gene_expr']
            num_cells = processed_batch['num_cells']
            real_images = batch['image']
            patch_ids = batch['patch_id']
            
            # Generate images with diffusion
            generated_images = generate_images_with_diffusion(
                model=model,
                diffusion=diffusion,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                num_cells=num_cells,
                is_multi_cell=True,
                method="ddim"  # Use DDIM for faster sampling
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

    # Cleanup DDP
    if args.use_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()