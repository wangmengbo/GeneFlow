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
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.single_model import RNAtoHnEModel
from rectified.rectified_flow import RectifiedFlow
from src.utils import (
	setup_parser, parse_adata, analyze_gene_importance, get_hash,
    setup_ddp, cleanup_ddp, load_checkpoint, save_checkpoint,
    save_checkpoint_with_spatial_info, setup_logging
)
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import (
    CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn, 
    load_preprocessed_hest1k_singlecell_data, 
    OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn, 
    FastSeparatePatchDataset, fast_separate_patch_collate_fn
    )
from rectified.rectified_train import train_with_rectified_flow, generate_images_with_rectified_flow


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RNA to H&E cell image generator with Rectified Flow.")
    
    # Training-specific arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for training.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel training.')
    parser.add_argument('--prefix', type=str, default='geneflow', help='Prefix for wandb run name.')
    parser.add_argument('--no_wandb', action='store_true', default=True, help='Disable wandb logging.')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb run ID for resuming.')
    parser.add_argument('--log_interval_pct', type=float, default=1.0, help='Log progress every N percent of batches')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to specific checkpoint to resume from.')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from latest checkpoint.')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small subset of data.')
    parser.add_argument('--debug_samples', type=int, default=1000, help='Number of samples to use in debug mode.')
    parser.add_argument('--max_checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep.')
    
    # Spatial loss arguments
    parser.add_argument('--use_spatial_loss', action='store_true', help='Use spatial graph loss for multi-cell training')
    parser.add_argument('--spatial_loss_method', type=str, default='simple', choices=['simple', 'segmentation'],
                       help='Method for spatial loss')
    parser.add_argument('--force_spatial_from_resume', action='store_true',
                       help='Force spatial loss to start immediately when resuming')
    parser.add_argument('--spatial_loss_weight', type=float, default=0.1,
                       help='Weight for spatial graph loss term')
    parser.add_argument('--spatial_loss_k_neighbors', type=int, default=5,
                       help='Number of neighbors for spatial graph construction')
    parser.add_argument('--spatial_loss_start_epoch', type=int, default=None,
                       help='Epoch to start spatial loss')
    parser.add_argument('--spatial_loss_start_val_loss', type=float, default=None,
                       help='Start spatial loss when val_loss drops below this threshold')
    parser.add_argument('--spatial_loss_warmup_epochs', type=int, default=5,
                       help='Number of epochs to warmup spatial loss weight')
    parser.add_argument('--spatial_patience', type=int, default=None,
                       help='Early stopping patience after spatial loss starts')
    
    parser = setup_parser(parser)
    args = parser.parse_args()

    if args.use_ddp:
        rank, local_rank, world_size, device = setup_ddp()
        # Keep batch_size as specified - it applies to each GPU
        per_gpu_batch_size = args.batch_size
        total_batch_size = args.batch_size * world_size
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        per_gpu_batch_size = args.batch_size
        total_batch_size = args.batch_size

    logger = setup_logging(rank)
    
    if rank == 0:
        logger.info(f"args={args}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Using device: {device}, Rank: {rank}, World size: {world_size}")
        if args.use_ddp:
            logger.info(f"Per-GPU batch size: {per_gpu_batch_size}, Total effective batch size: {total_batch_size}")
        else:
            logger.info(f"Batch size: {per_gpu_batch_size}")

        if not args.no_wandb:
            if args.wandb_id:
                run_id = args.wandb_id
                resume_mode = "must"
            else:
                run_id = None
                resume_mode = "allow"
                
            wandb.init(
                project="GeneFlow",
                entity="cell-image",
                name=f"{args.prefix}_{get_hash(args)}",
                id=run_id,
                config=vars(args),
                dir=args.output_dir,
                resume=resume_mode,
            )
        
    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    expr_df = None
    gene_names = None
    
    # Data loading logic (same as original)
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

    # Dataset creation logic (same as original)
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
    else:
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
            if rank == 0:
                logger.info(f"Loading patch-to-cell mapping from {args.patch_cell_mapping}")
            with open(args.patch_cell_mapping, "r") as f:
                patch_to_cells = json.load(f)
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

    if args.debug:
        if rank == 0:
            logger.info(f"DEBUG MODE: Using only {args.debug_samples} samples")
        # Create a subset of the full dataset
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.debug_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        if rank == 0:
            logger.info(f"Debug dataset size: {len(dataset)}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
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

    # Model initialization
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
        concat_mask=args.concat_mask,
        relation_rank=args.relation_rank,
        use_multi_head_attention=args.use_multi_head_attention,
        use_feature_gating=args.use_feature_gating,
        use_residual_blocks=args.use_residual_blocks,
        use_layer_norm=args.use_layer_norm,
        use_gene_relations=args.use_gene_relations,
    )

    if args.model_type == 'single':
        if rank == 0:
            logger.info("Initializing single-cell model")
        model_constructor_args = model_constructor_args | dict(
            use_gene_attention=args.use_gene_attention,
        )
        model = RNAtoHnEModel(**model_constructor_args)
    else:
        if rank == 0:
            logger.info("Initializing multi-cell model")
        model = MultiCellRNAtoHnEModel(**model_constructor_args, num_aggregation_heads=args.num_aggregation_heads)

    # Move model to device
    model.to(device)

    # Wrap model with DDP if using distributed training
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")

    if rank == 0:
        logger.info(f"Model initialized with gene dimension: {gene_dim}")

    # Rectified flow initialization
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    if rank == 0:
        logger.info("Initialized rectified flow")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    # Checkpoint path
    checkpoint_path = os.path.join(args.output_dir, "training_checkpoint.pt")
    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoint if it exists (only on rank 0, then broadcast)
    checkpoint_to_load = None
    checkpoint_state = None

    if rank == 0:
        if args.resume_from is not None:
            if os.path.exists(args.resume_from):
                checkpoint_to_load = args.resume_from
                logger.info(f"Resuming from specified checkpoint: {checkpoint_to_load}")
            else:
                logger.error(f"Specified checkpoint not found: {args.resume_from}")
                raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        elif args.auto_resume:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            
            if os.path.exists(latest_checkpoint):
                checkpoint_to_load = latest_checkpoint
                logger.info(f"Auto-resuming from latest checkpoint: {checkpoint_to_load}")
            else:
                logger.info("--auto_resume set but no checkpoint found. Starting from scratch.")
        else:
            logger.info("Starting training from scratch (use --auto_resume to resume automatically).")
        
        # Load the checkpoint if one was found
        if checkpoint_to_load is not None:
            checkpoint_state = torch.load(checkpoint_to_load, weights_only=False)
            start_epoch = checkpoint_state['epoch'] + 1  # Start from next epoch
            best_val_loss = checkpoint_state.get('best_val_loss', float('inf'))
            
            # Load model state
            model_state_dict = checkpoint_state.get('model_state_dict', checkpoint_state.get('model'))
            if isinstance(model, DDP):
                if not any(key.startswith('module.') for key in model_state_dict.keys()):
                    model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            else:
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            model.load_state_dict(model_state_dict)
            
            # Load optimizer and scheduler
            if 'optimizer_state_dict' in checkpoint_state:
                optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint_state:
                lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state_dict'])
            
            # Check spatial loss info
            spatial_was_enabled = checkpoint_state.get('spatial_loss_enabled', False)
            spatial_prev_start = checkpoint_state.get('spatial_loss_start_epoch', None)
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint_state['epoch']} (val_loss: {best_val_loss:.4f})")
            logger.info(f"Will resume training from epoch {start_epoch} (displayed as 'Epoch {start_epoch+1}')")
            if spatial_was_enabled:
                logger.info(f"Previous training had spatial loss enabled from epoch {spatial_prev_start}")
            else:
                logger.info("Previous training did not use spatial loss")
            
            if args.use_spatial_loss and not spatial_was_enabled:
                logger.info("Will enable spatial loss in this resumed training")
                if args.force_spatial_from_resume:
                    logger.info("  -> Spatial loss will start IMMEDIATELY (force_spatial_from_resume=True)")
                elif args.spatial_loss_start_epoch:
                    logger.info(f"  -> Spatial loss will start at epoch {args.spatial_loss_start_epoch}")
                else:
                    logger.info(f"  -> Spatial loss will start at default epoch (70% of total)")

    # Broadcast checkpoint info to all ranks
    if args.use_ddp:
        checkpoint_info = torch.tensor([start_epoch, best_val_loss], device=device)
        dist.broadcast(checkpoint_info, src=0)
        start_epoch, best_val_loss = int(checkpoint_info[0].item()), float(checkpoint_info[1].item())

    # Train model
    resuming_same_dir = False
    if checkpoint_to_load is not None:
        checkpoint_dir_parent = os.path.dirname(os.path.dirname(checkpoint_to_load))
        resuming_same_dir = os.path.samefile(checkpoint_dir_parent, args.output_dir)
    
    if checkpoint_to_load is not None and not resuming_same_dir and rank == 0:
        # Only copy if resuming from a DIFFERENT directory
        logger.info(f"Resuming from different directory, copying checkpoint to: {args.output_dir}")
        
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        spatial_suffix = "_spatial" if spatial_was_enabled else ""
        initial_checkpoint_name = f"checkpoint_after_epoch_{checkpoint_state['epoch']}_loss_{best_val_loss:.4f}_resumed{spatial_suffix}.pt"
        initial_checkpoint_file = os.path.join(checkpoint_dir, initial_checkpoint_name)
        
        initial_checkpoint_state = {
            'epoch': checkpoint_state['epoch'],
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': checkpoint_state.get('train_loss', 0.0),
            'val_loss': best_val_loss,
            'spatial_loss_enabled': spatial_was_enabled,
            'spatial_loss_start_epoch': spatial_prev_start,
        }
        torch.save(initial_checkpoint_state, initial_checkpoint_file)
        
        # Create initial symlinks
        latest_link = os.path.join(checkpoint_dir, f"latest_checkpoint{spatial_suffix}.pt")
        best_link = os.path.join(checkpoint_dir, f"best_checkpoint{spatial_suffix}.pt")
        
        for link_path in [latest_link, best_link]:
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            try:
                os.symlink(initial_checkpoint_name, link_path)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(initial_checkpoint_file, link_path)
        
        logger.info(f"Initial checkpoint saved to new output directory")
    elif checkpoint_to_load is not None and resuming_same_dir and rank == 0:
        logger.info("Resuming from same directory, using existing checkpoints")

    if args.use_ddp:
        checkpoint_info = torch.tensor([start_epoch, best_val_loss], device=device)
        dist.broadcast(checkpoint_info, src=0)
        start_epoch, best_val_loss = int(checkpoint_info[0].item()), float(checkpoint_info[1].item())

    # Train model
    best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_rna_to_hne_model_rectified.pt")

    if not getattr(args, 'only_inference', False):
        train_losses, val_losses = train_with_rectified_flow(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            rectified_flow=rectified_flow,
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
            lr_scheduler=lr_scheduler,
            checkpoint_path=checkpoint_path,
            save_checkpoint_fn=save_checkpoint,
            rank=rank,
            world_size=world_size,
            use_ddp=args.use_ddp,
            no_wandb=args.no_wandb,
            log_interval_pct=args.log_interval_pct,
            max_checkpoints=args.max_checkpoints,
            use_spatial_loss=args.use_spatial_loss,
            spatial_loss_method=args.spatial_loss_method,
            spatial_loss_weight=args.spatial_loss_weight,
            spatial_loss_k_neighbors=args.spatial_loss_k_neighbors,
            spatial_loss_start_epoch=args.spatial_loss_start_epoch,
            spatial_loss_start_val_loss=args.spatial_loss_start_val_loss,
            spatial_loss_warmup_epochs=args.spatial_loss_warmup_epochs,
            spatial_patience=args.spatial_patience if args.spatial_patience is not None else args.patience,
            force_spatial_loss_from_resume=args.force_spatial_from_resume,
            resumed_checkpoint_state=checkpoint_state,
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
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        best_checkpoint_link = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        
        if not os.path.exists(best_checkpoint_link):
            # Fallback to spatial checkpoint if regular one doesn't exist
            best_checkpoint_link = os.path.join(checkpoint_dir, "best_checkpoint_spatial.pt")
        
        logger.info(f"Loading best model from {best_checkpoint_link}")
        checkpoint = torch.load(best_checkpoint_link, weights_only=True)
        
        # Handle both checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            raise KeyError(f"Checkpoint missing model weights. Available keys: {checkpoint.keys()}")
        
        if isinstance(model, DDP):
            if not any(key.startswith('module.') for key in model_state_dict.keys()):
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        else:
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        model.load_state_dict(model_state_dict)

        # Generate images from validation set (only on rank 0)
        logger.info("Generating images from validation set...")
        num_vis_samples = min(10, len(val_dataset))
        vis_indices = torch.randperm(len(val_dataset))[:num_vis_samples]
        vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)

        # Create visualization DataLoader
        if args.model_type == 'multi':
            if args.hest1k_xenium_fast_dir is not None:
                vis_collate_fn = fast_separate_patch_collate_fn
            elif args.hest1k_xenium_dir is not None:
                vis_collate_fn = multi_sample_hest_xenium_collate_fn
            else:
                vis_collate_fn = patch_collate_fn
                
            vis_loader = DataLoader(vis_dataset, batch_size=num_vis_samples, shuffle=False,
                                collate_fn=vis_collate_fn)
        else:
            vis_loader = DataLoader(vis_dataset, batch_size=num_vis_samples, shuffle=False)

        batch = next(iter(vis_loader))

        # Generate images
        if args.model_type == 'single':
            gene_expr = batch['gene_expr'].to(device)
            real_images = batch['image']
            cell_ids = batch['cell_id']
            gene_mask = batch.get('gene_mask', None)
            if gene_mask is not None:
                gene_mask = gene_mask.to(device)
            generated_images = generate_images_with_rectified_flow(
                model=model,
                rectified_flow=rectified_flow,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                gene_mask=gene_mask,
                is_multi_cell=False
            )
        else:
            processed_batch = prepare_multicell_batch(batch, device)
            gene_expr = processed_batch['gene_expr']
            logger.info(f"Gene expression shape: {gene_expr.shape}")
            
            num_cells = processed_batch['num_cells']
            real_images = batch['image']
            patch_ids = batch['patch_id']
            generated_images = generate_images_with_rectified_flow(
                model=model,
                rectified_flow=rectified_flow,
                gene_expr=gene_expr,
                device=device,
                num_steps=args.gen_steps,
                num_cells=num_cells,
                is_multi_cell=True
            )

        # Save results (rest of visualization code remains the same)
        os.makedirs(os.path.join(args.output_dir, "generated_images"), exist_ok=True)
        num_channels = args.img_channels
        num_extra_channels = max(0, num_channels - 3)
        num_rows = 2 + (2 * num_extra_channels)
        fig, axes = plt.subplots(num_rows, num_vis_samples, figsize=(3*num_vis_samples, 2*num_rows))

        if num_vis_samples == 1:
            axes = np.expand_dims(axes, axis=1)
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        display_ids = cell_ids if args.model_type == 'single' else patch_ids

        for i in range(num_vis_samples):
            real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
            gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)

            axes[0, i].imshow(real_img[:,:,:3])
            axes[0, i].set_title(f"Real RGB: {display_ids[i]}")
            axes[0, i].axis('off')

            axes[1, i].imshow(gen_img[:,:,:3])
            axes[1, i].set_title("Generated RGB")
            axes[1, i].axis('off')

            plt.imsave(os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_real_rgb.png"), real_img[:,:,:3])
            plt.imsave(os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_gen_rgb.png"), gen_img[:,:,:3])

            for c in range(3, num_channels):
                real_row_idx = 2 + (2 * (c - 3))
                gen_row_idx = 3 + (2 * (c - 3))
                axes[real_row_idx, i].imshow(real_img[:,:,c], cmap='gray')
                axes[real_row_idx, i].set_title(f"Real Ch{c}")
                axes[real_row_idx, i].axis('off')

                axes[gen_row_idx, i].imshow(gen_img[:,:,c], cmap='gray')
                axes[gen_row_idx, i].set_title(f"Gen Ch{c}")
                axes[gen_row_idx, i].axis('off')

                plt.imsave(os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_real_ch{c}.png"), real_img[:,:,c], cmap='gray')
                plt.imsave(os.path.join(args.output_dir, "generated_images", f"{display_ids[i]}_gen_ch{c}.png"), gen_img[:,:,c], cmap='gray')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "generation_results.png"))
        logger.info(f"Results saved to {args.output_dir}")

        # Gene importance analysis (only on rank 0)
        if args.model_type == 'single' and gene_names is not None:
            importance_output_path = os.path.join(args.output_dir, "gene_importance_scores.csv")
            analyze_gene_importance(
                model=model,
                data_loader=val_loader,
                rectified_flow=rectified_flow,
                device=device,
                gene_names=gene_names,
                output_path=importance_output_path,
                timesteps_to_analyze=getattr(args, 'analysis_timesteps', None),
                num_batches_to_analyze=getattr(args, 'analysis_batches', None),
            )
            logger.info(f"Gene importance scores saved to {importance_output_path}")

    if rank == 0 and not args.no_wandb:
        wandb.finish()

    if args.use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()

