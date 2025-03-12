import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import hashlib
import json
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


def get_stable_run_id(args):
    """Generate stable ID based on core experiment params, not training params"""
    # Only include params that define the experiment identity
    stable_params = {
        'model_type': args.model_type,
        'samples': args.hest1k_xenium_samples,
        'img_size': args.img_size,
        'img_channels': args.img_channels,
        'seed': args.seed,
        # Add other params that define the experiment
    }
    import hashlib
    stable_str = json.dumps(stable_params, sort_keys=True)
    return hashlib.md5(stable_str.encode()).hexdigest()[:8]
    

def save_checkpoint(state, filename, rank, is_best=False, keep_latest_pointer=True):
    """Save checkpoint only on rank 0 with optional pointers"""
    if rank == 0:
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Save the actual checkpoint
        torch.save(state, filename)
        logger.info(f"Checkpoint saved: {filename}")
        
        # Create pointer to latest checkpoint
        if keep_latest_pointer:
            checkpoint_dir = os.path.dirname(filename)
            latest_link = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            
            # Remove old symlink/file if exists
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.remove(latest_link)
            
            # Create symlink (or copy if symlink fails on Windows)
            try:
                os.symlink(os.path.basename(filename), latest_link)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(filename, latest_link)
            
            logger.info(f"Latest checkpoint pointer updated: {latest_link}")
        
        # Create pointer to best checkpoint
        if is_best:
            checkpoint_dir = os.path.dirname(filename)
            best_link = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            
            # Remove old symlink/file if exists
            if os.path.exists(best_link) or os.path.islink(best_link):
                os.remove(best_link)
            
            # Create symlink (or copy if symlink fails)
            try:
                os.symlink(os.path.basename(filename), best_link)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(filename, best_link)
            
            logger.info(f"Best checkpoint pointer updated: {best_link}")


def save_checkpoint_with_spatial_info(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    lr_scheduler,
    best_val_loss,
    train_loss,
    val_loss,
    scaler=None,
    spatial_loss_enabled=False,
    spatial_loss_start_epoch=None
):
    """Save checkpoint with spatial loss information"""
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint_state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'spatial_loss_enabled': spatial_loss_enabled,
        'spatial_loss_start_epoch': spatial_loss_start_epoch,
    }
    if scaler is not None:
        checkpoint_state['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint_state, checkpoint_path)


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
        

def get_hash(namespace):
    """
    Generate an 8-character hash from argparse Namespace object.
    
    Args:
        namespace: argparse.Namespace object from parse_args()
        
    Returns:
        str: 8-character hexadecimal hash
    """
    # Convert namespace to dict and create hash
    data = json.dumps(vars(namespace), sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:8]


def setup_parser(parser=None):
    """
    Add common arguments used across train, generate, and evaluate scripts.
    Call this AFTER adding script-specific arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="RNA to H&E image generation tool")

    arch_group = parser.add_argument_group('Model Architecture')
    arch_group.add_argument('--use_gene_attention', action='store_true', default=True, 
                           help='Use gene attention mechanism (single-cell only)')
    arch_group.add_argument('--no_gene_attention', dest='use_gene_attention', action='store_false', 
                           help='Disable gene attention mechanism')
    arch_group.add_argument('--use_multi_head_attention', action='store_true', default=True, 
                           help='Use multi-head attention')
    arch_group.add_argument('--no_multi_head_attention', dest='use_multi_head_attention', action='store_false', 
                           help='Disable multi-head attention')
    arch_group.add_argument('--use_feature_gating', action='store_true', default=True, 
                           help='Use feature gating')
    arch_group.add_argument('--no_feature_gating', dest='use_feature_gating', action='store_false', 
                           help='Disable feature gating')
    arch_group.add_argument('--use_residual_blocks', action='store_true', default=True, 
                           help='Use residual blocks')
    arch_group.add_argument('--no_residual_blocks', dest='use_residual_blocks', action='store_false', 
                           help='Disable residual blocks')
    arch_group.add_argument('--use_layer_norm', action='store_true', default=True, 
                           help='Use layer normalization')
    arch_group.add_argument('--no_layer_norm', dest='use_layer_norm', action='store_false', 
                           help='Disable layer normalization')
    arch_group.add_argument('--use_gene_relations', action='store_true', default=True, 
                           help='Use gene relations')
    arch_group.add_argument('--no_gene_relations', dest='use_gene_relations', action='store_false', 
                           help='Disable gene relations')
    arch_group.add_argument('--concat_mask', action='store_true', default=False, 
                           help='Concatenate mask to the input of the RNA encoder')
    arch_group.add_argument('--relation_rank', type=int, default=50, 
                           help='Rank K for low-rank factorization in gene relation network (default: 50)')
    arch_group.add_argument('--num_aggregation_heads', type=int, default=4, 
                           help='Number of heads for cell aggregation in MultiCellRNAEncoder (multi-cell only, default: 4)')

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi',
                            help='Type of model to use: single-cell or multi-cell')
    model_group.add_argument('--img_size', type=int, default=256, 
                            help='Size of the generated images')
    model_group.add_argument('--img_channels', type=int, default=3, 
                            help='Number of image channels (3 for RGB, 4 for RGB+aux)')
    model_group.add_argument('--normalize_aux', action='store_true', 
                            help='Normalize auxiliary channels')
    model_group.add_argument('--gen_steps', type=int, default=100, 
                            help='Number of steps for solver during generation')
    model_group.add_argument('--seed', type=int, default=42, 
                            help='Random seed for reproducibility')

    data_group = parser.add_argument_group('Data Loading')
    
    data_group.add_argument('--adata', type=str, default=None,
                           help='Path to the AnnData object')
    data_group.add_argument('--layer', type=str, default=None, 
                           help='Layer to use for the AnnData object')
    data_group.add_argument('--cell_type', type=str, nargs='*', default=None,
                           help='Cell types to include')
    data_group.add_argument('--exclude_cell_type', type=str, nargs='*', default=None,
                           help='Cell types to exclude')
    data_group.add_argument('--cell_type_label', type=str, default='cell_type',
                           help='Column name for cell type labels')
    data_group.add_argument('--min_total_counts', type=int, default=0,
                           help='Minimum total counts for filtering')
    data_group.add_argument('--max_total_counts', type=int, default=np.inf,
                           help='Maximum total counts for filtering')
    data_group.add_argument('--min_total_pct', type=float, default=0.0,
                           help='Minimum total counts percentile')
    data_group.add_argument('--max_total_pct', type=float, default=1.0,
                           help='Maximum total counts percentile')
    data_group.add_argument('--use_full_gene_list', action='store_true', default=False, 
                           help='Use the full gene list instead of top highly variable genes')
    data_group.add_argument('--gene_symbols', type=str, default=None, 
                           help='Path to the gene symbol list')
    data_group.add_argument('--missing_gene_symbols', type=str, default=None, 
                           help='Path to a file containing missing gene symbols, one per line')
    
    # CSV/JSON loading (deprecated but kept for backward compatibility)
    data_group.add_argument('--gene_expr', type=str, default=None,
                           help='Path to gene expression CSV file (deprecated)')
    data_group.add_argument('--image_paths', type=str, default=None,
                           help='Path to JSON file with image paths (deprecated)')
    data_group.add_argument('--patch_image_paths', type=str, default=None,
                           help='Path to JSON file with patch paths (deprecated)')
    data_group.add_argument('--patch_cell_mapping', type=str, default=None,
                           help='Path to JSON file with mapping paths (deprecated)')
    
    # HEST-1k data loading
    data_group.add_argument('--hest1k_sid', type=str, nargs='*', default=None,
                           help='HEST-1k sample ID for direct loading')
    data_group.add_argument('--hest1k_base_dir', type=str, default=None,
                           help='Base directory for HEST-1k data')
    data_group.add_argument('--hest1k_xenium_dir', type=str, default=None,
                           help='Directory for HEST-1k Xenium AnnData files')
    data_group.add_argument('--hest1k_xenium_metadata', type=str, default=None,
                           help='Metadata CSV for HEST-1k Xenium data')
    data_group.add_argument('--hest1k_xenium_samples', type=str, nargs='*', default=None,
                           help='Specific Xenium sample IDs to use')
    data_group.add_argument('--hest1k_xenium_fast_dir', type=str, default=None,
                           help='Directory for reformatted fast-loading HEST-1k Xenium patch data')
    data_group.add_argument('--num_dataloader_workers', type=int, default=4,
                           help='Number of workers for data loading')
    
    output_group = parser.add_argument_group('Output and Evaluation')
    output_group.add_argument('--output_dir', type=str, default='output',
                             help='Directory to save outputs')
    output_group.add_argument('--output_name_prefix', type=str, default='',
                             help='Prefix for the output evaluation files')
    output_group.add_argument('--cell_id_to_generate', type=str, default=None, nargs='*',
                             help='Cell IDs to generate images for. If not provided, all cells will be used')
    
    inference_group = parser.add_argument_group('Inference and Analysis')
    inference_group.add_argument('--only_inference', action='store_true', default=False,
                                help='Only run inference')
    inference_group.add_argument('--only_importance_analysis', action='store_true', default=False,
                                help='Only run gene importance analysis')
    inference_group.add_argument('--analysis_timesteps', type=float, nargs='+', 
                                default=[0.1, 0.3, 0.5, 0.7, 0.9],
                                help='List of timesteps (0 to 1) to use for importance analysis')
    inference_group.add_argument('--analysis_batches', type=int, default=10,
                                help='Number of validation batches to use for importance analysis')
    inference_group.add_argument('--nsamples_test', type=int, default=-1,
                                help='Number of batches to use for testing')
    
    stain_group = parser.add_argument_group('Stain Normalization')
    stain_group.add_argument('--enable_stain_normalization', action='store_true',
                            help='Enable stain normalization of generated images to real images')
    stain_group.add_argument('--stain_normalization_method', type=str, default='skimage_hist_match',
                            choices=['skimage_hist_match', 'none'],
                            help='Stain normalization method')
    
    return parser

def parse_adata(args=None, 
                adata=None,
                layer=None,
                cell_type=None, 
                exclude_cell_type=None,
                cell_type_label=None, 
                min_total_counts=None, 
                max_total_counts=None, 
                min_total_pct=None, 
                max_total_pct=None,
                # use_full_gene_list=None,
                gene_symbols=None,
                missing_gene_symbols=None,
                concat_mask=None,
                nsamples_test=None,
                ):
    # clean arguments
    if args is not None:
        if adata is None and args.adata is not None:
            adata = args.adata
        if layer is None and args.layer is not None:
            layer = args.layer
        if cell_type is None and args.cell_type is not None:
            cell_type = args.cell_type
        if exclude_cell_type is None and args.exclude_cell_type is not None:
            exclude_cell_type = args.exclude_cell_type
        if cell_type_label is None:
            cell_type_label = args.cell_type_label
        if min_total_counts is None and args.min_total_counts is not None:
            min_total_counts = args.min_total_counts
        if max_total_counts is None and args.max_total_counts is not None:
            max_total_counts = args.max_total_counts
        if min_total_pct is None and args.min_total_pct is not None:
            min_total_pct = args.min_total_pct
        if max_total_pct is None and args.max_total_pct is not None:
            max_total_pct = args.max_total_pct
        if gene_symbols is None:
            gene_symbols = args.gene_symbols
        if missing_gene_symbols is None:
            missing_gene_symbols = args.missing_gene_symbols
        if concat_mask is None:
            concat_mask = args.concat_mask
        if nsamples_test is None:
            nsamples_test = args.nsamples_test
    
    # parse adata
    if type(adata) is str:
        adata = sc.read_h5ad(adata)
        logger.info(f"Loaded AnnData object from {adata}")
        logger.info(f"AnnData object has {adata.n_obs} cells and {adata.n_vars} genes")
    
    if layer is not None:
        adata.X = adata.layers[layer]

    if cell_type is not None:
        logger.info(f"Filtering cells with cell type {cell_type}")
        adata = adata[adata.obs[cell_type_label].isin(cell_type)]
        logger.info(f"{len(adata)} cells with cell type {cell_type} passed the filter")

    if exclude_cell_type is not None:
        logger.info(f"Filtering cells other than cell type {exclude_cell_type}")
        adata = adata[~adata.obs[cell_type_label].isin(exclude_cell_type)]
        logger.info(f"{len(adata)} cells other than cell type {exclude_cell_type} passed the filter")

    if min_total_counts is not None and min_total_counts > 0:
        logger.info(f"Filtering cells with total counts < {min_total_counts}")
        adata = adata[adata.obs["total_counts"] >= min_total_counts]
        logger.info(f"{len(adata)} cells with total counts > {min_total_counts} passed the filter")
    
    if max_total_counts is not None and max_total_counts < np.inf:
        logger.info(f"Filtering cells with total counts > {max_total_counts}")
        adata = adata[adata.obs["total_counts"] <= max_total_counts]
        logger.info(f"{len(adata)} cells with total counts < {max_total_counts} passed the filter")
    
    if min_total_pct is not None and min_total_pct > 0.0:
        logger.info(f"Filtering cells with total pct < {min_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], min_total_pct * 100)
        adata = adata[adata.obs["total_counts"] >= threshold]
    
    if max_total_pct is not None and max_total_pct < 1.0:
        logger.info(f"Filtering cells with total pct > {max_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], max_total_pct * 100)
        adata = adata[adata.obs["total_counts"] <= threshold]
        logger.info(f"{len(adata)} cells with total pct < {max_total_pct * 100}% passed the filter")

    if missing_gene_symbols is not None and os.path.isfile(missing_gene_symbols):
        missing_gene_symbols = pd.read_csv(missing_gene_symbols, header=None)[0].tolist()
        logger.info(f"Loaded {len(missing_gene_symbols)} missing gene symbols from {args.missing_gene_symbols}")
    else:
        missing_gene_symbols = set()

    if gene_symbols is not None and os.path.isfile(gene_symbols):
        gene_symbols = pd.read_csv(gene_symbols, header=None)[0].tolist()

    if nsamples_test is not None and nsamples_test > 0:
        logger.info(f"Subsampling {nsamples_test} cells for testing")
        sc.pp.subsample(adata, n_obs=nsamples_test)
        logger.info(f"Subsampled {len(adata)} cells for testing")

    ngenes = adata.n_vars
    genes = adata.var_names.tolist()
    expr = adata.to_df()
    # mask = None
    # if use_full_gene_list:
    if gene_symbols is not None and len(gene_symbols) > 0:
        ngenes = len(gene_symbols)
        genes = gene_symbols
        expr = pd.DataFrame(np.zeros((adata.n_obs, ngenes)), index=adata.obs_names, columns=gene_symbols)
        expr.update(adata.to_df())
        missing_gene_symbols = list(set(missing_gene_symbols) | (set(gene_symbols) - set(adata.var_names)))
    
    return expr, missing_gene_symbols
    

def analyze_gene_importance(
    model,
    data_loader,
    rectified_flow, # Pass RectifiedFlow object if needed for sampling x_t
    device,
    gene_names,
    output_path,
    timesteps_to_analyze=[0.1, 0.5, 0.9],
    num_batches_to_analyze=np.inf, # Limit number of batches for efficiency
):
    """
    Performs gradient-based gene importance analysis.

    Args:
        model: The trained RNAtoHnEModel.
        data_loader: DataLoader (e.g., validation loader) to get RNA samples.
        rectified_flow: RectifiedFlow instance (optional, for sampling x_t).
        device: Computation device.
        gene_names: List of gene names corresponding to rna_expr dimensions.
        output_path: Path to save the CSV results.
        timesteps_to_analyze: List of timesteps (0 to 1) to analyze.
        num_batches_to_analyze: Max number of batches to process from data_loader.
    """
    logger.info("Starting gradient-based gene importance analysis...")
    model.eval() # Ensure model is in eval mode

    # Initialize tensor to store cumulative absolute gradients for each gene
    # Use float64 for accumulation to prevent potential overflow/precision issues
    gene_gradients_sum = torch.zeros(model.rna_dim, dtype=torch.float64, device=device)
    num_samples_processed = 0
    batches_processed = 0

    timesteps_tensor = torch.tensor(timesteps_to_analyze, device=device)

    # Get expected image shape once
    try:
        # Attempt to get shape from a sample batch image if possible
        sample_batch = next(iter(data_loader))
        _, C, H, W = sample_batch['image'].shape
        logger.info(f"Inferred image shape: C={C}, H={H}, W={W}")
    except Exception:
        # Fallback to model config
        C, H, W = model.img_channels, model.img_size, model.img_size
        logger.warning(f"Could not infer image shape from data, using model config: C={C}, H={H}, W={W}")

    # Iterate through data loader batches
    pbar_batches = tqdm(data_loader, total=min(num_batches_to_analyze, len(data_loader)), desc="Analyzing Batches")
    for batch in pbar_batches:
        if batches_processed >= num_batches_to_analyze:
            break

        rna_expr_batch = batch['gene_expr'].to(device)
        current_batch_size = rna_expr_batch.shape[0]

        # Enable gradient calculation for this specific RNA input batch
        rna_expr_batch.requires_grad_(True)

        # Generate noise once per batch (or sample x_t if preferred)
        # Using same noise across timesteps for this batch for simplicity
        x_t_noise = torch.randn(current_batch_size, C, H, W, device=device)

        # Iterate through specified timesteps
        for t_val in timesteps_tensor:
            t_batch = torch.full((current_batch_size,), t_val.item(), device=device)

            # --- Gradient Calculation ---
            with torch.set_grad_enabled(True):
                # Zero gradients before calculation
                model.zero_grad()
                if rna_expr_batch.grad is not None:
                    rna_expr_batch.grad.zero_()

                # Forward pass
                v_pred = model(x_t_noise, t_batch, rna_expr_batch)

                # Scalar output: L2 norm squared of velocity, summed over batch
                scalar_output = torch.sum(v_pred**2)

                # Backward pass
                scalar_output.backward()

            if rna_expr_batch.grad is not None:
                # Sum absolute gradients across the batch dimension for this timestep
                # Move to float64 before summing potentially large numbers
                batch_gene_grads = rna_expr_batch.grad.abs().sum(dim=0).to(torch.float64)
                gene_gradients_sum += batch_gene_grads
                num_samples_processed += current_batch_size # Increment by samples in this timestep analysis
            else:
                logger.warning(f"No gradient computed for rna_expr_batch at t={t_val.item()} in batch {batches_processed}")

        # Detach the input batch after processing all timesteps for it
        rna_expr_batch = rna_expr_batch.detach()
        batches_processed += 1
        pbar_batches.set_postfix({"Samples processed": num_samples_processed})

    pbar_batches.close()

    if num_samples_processed > 0:
        # Average the summed absolute gradients over all samples processed (batches * timesteps)
        avg_gene_importance = (gene_gradients_sum / num_samples_processed).cpu().numpy()

        # Create DataFrame and save results
        importance_df = pd.DataFrame({
            'gene_name': gene_names,
            'importance_score': avg_gene_importance
        })
        importance_df = importance_df.sort_values(by='importance_score', ascending=False)

        importance_df.to_csv(output_path, index=False)
        logger.info(f"Gene importance scores saved to {output_path}")
        logger.info("Top 5 important genes:")
        logger.info(importance_df.head(5))
    else:
        logger.warning("No samples were processed for gradient analysis. Importance scores not calculated.")
    
    return importance_df


def analyze_gene_importance_diffusion(
    model,
    data_loader,
    diffusion,
    device,
    gene_names,
    output_path="gene_importance_scores.csv",
    timesteps_to_analyze=None,
    num_batches_to_analyze=None
):
    """
    Analyze gene importance by computing gradient-based importance scores
    
    Args:
        model: The RNA to H&E model
        data_loader: DataLoader for image-gene expression pairs
        diffusion: The diffusion object
        device: Computation device
        gene_names: List of gene names corresponding to input dimensions
        output_path: Path to save the importance scores
        timesteps_to_analyze: List of specific timesteps to analyze (default: 4 evenly spaced timesteps)
        num_batches_to_analyze: Number of batches to use for analysis (default: 5)
    """
    logger.info("Analyzing gene importance...")
    
    # Set model to eval mode but enable gradients for feature importance
    model.eval()
    
    # Get the gene dimension from the model
    gene_dim = model.rna_dim
    
    # Initialize array to store importance scores
    importance_scores = torch.zeros(gene_dim, device=device)
    total_samples = 0
    
    # Set default timesteps if not provided (4 evenly spaced timesteps)
    if timesteps_to_analyze is None:
        timesteps_to_analyze = [
            0,  # Beginning of diffusion (mostly noise)
            diffusion.timesteps // 4,  # 25% through diffusion
            diffusion.timesteps // 2,  # 50% through diffusion
            3 * diffusion.timesteps // 4,  # 75% through diffusion
        ]
    
    # Set default number of batches if not provided
    if num_batches_to_analyze is None:
        num_batches_to_analyze = 5
    
    # Get a subset of batches
    batch_count = 0
    
    for batch in data_loader:
        if batch_count >= num_batches_to_analyze:
            break
            
        gene_expr = batch['gene_expr'].to(device)
        target_images = batch['image'].to(device)
        
        # Skip batches with no gradients (in case of NaNs or other issues)
        if torch.isnan(gene_expr).any() or torch.isnan(target_images).any():
            logger.warning(f"Skipping batch {batch_count} due to NaN values")
            continue
        
        batch_size = gene_expr.shape[0]
        
        # Analyze each specified timestep
        for t_step in timesteps_to_analyze:
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            
            # Add noise according to timestep
            noisy_images, target_noise = diffusion.q_sample(target_images, t)
            
            # Enable gradients for this computation
            gene_expr.requires_grad_(True)
            
            # Get model prediction
            pred_noise = model(noisy_images, t, gene_expr)
            
            # Compute loss
            loss = torch.mean((pred_noise - target_noise) ** 2)
            
            # Compute gradients
            loss.backward()
            
            # Get gradients with respect to gene expressions
            gene_grads = gene_expr.grad.abs()
            
            # Average gradients across the batch
            avg_gene_grads = gene_grads.mean(dim=0)
            
            # Accumulate importance scores
            importance_scores += avg_gene_grads
            
            # Reset gradients
            gene_expr.grad.zero_()
            gene_expr.requires_grad_(False)
        
        total_samples += 1
        batch_count += 1
    
    # Average importance scores over all samples and timesteps
    if total_samples > 0:
        importance_scores /= (total_samples * len(timesteps_to_analyze))
    
    # Convert to numpy for saving
    importance_scores_np = importance_scores.cpu().numpy()
    
    # Create DataFrame with gene names and scores
    import pandas as pd
    gene_importance_df = pd.DataFrame({
        'gene': gene_names,
        'importance_score': importance_scores_np
    })
    
    # Sort by importance score
    gene_importance_df = gene_importance_df.sort_values('importance_score', ascending=False)
    
    # Save to CSV
    gene_importance_df.to_csv(output_path, index=False)
    
    logger.info(f"Gene importance analysis complete. Results saved to {output_path}")
    
    return gene_importance_df


def normalize_rgb(rgb_image):
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = ((rgb_image - np.min(rgb_image) + 1e-6) / (np.max(rgb_image) - np.min(rgb_image) + 1e-6))
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image


def normalize_aux(aux_image):
    aux_image = aux_image.astype(np.float32)
    aux_image = ((aux_image - np.min(aux_image) + 1e-6) / (np.max(aux_image) - np.min(aux_image) + 1e-6))
    aux_image = (aux_image * 255).astype(np.uint8)
    return aux_image


def manage_checkpoints(checkpoint_dir, max_checkpoints, rank, suffix=""):
    """
    Manage checkpoint files, keeping only the most recent max_checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        rank: Process rank (only rank 0 performs cleanup)
        suffix: Suffix to filter checkpoints (e.g., "_spatial" or "")
    """
    if rank != 0:
        return
    
    # Get all checkpoint files with the specified suffix (excluding symlinks)
    all_checkpoints = []
    for f in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, f)
        # Skip symlinks/pointers
        if os.path.islink(full_path):
            continue
        # Skip files without the suffix
        if suffix and not f.endswith(f"{suffix}.pt"):
            continue
        if not suffix and "_spatial.pt" in f:
            continue  # Skip spatial checkpoints when managing non-spatial
        # Only include actual checkpoint files
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt"):
            all_checkpoints.append(full_path)
    
    if len(all_checkpoints) <= max_checkpoints:
        return
    
    # Sort by modification time (oldest first)
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x))
    
    # Remove oldest checkpoints
    checkpoints_to_remove = all_checkpoints[:len(all_checkpoints) - max_checkpoints]
    for ckpt in checkpoints_to_remove:
        try:
            os.remove(ckpt)
            logger.info(f"Removed old checkpoint: {os.path.basename(ckpt)}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {ckpt}: {e}")