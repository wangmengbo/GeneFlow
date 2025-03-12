import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import inception_v3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sequoia_root = os.path.join(project_root, 'sequoia')
# sys.path.insert(0, sequoia_root)
# sys.path.insert(1, project_root)

from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata
from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation
from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import (
    CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn,
    load_preprocessed_hest1k_singlecell_data,
    OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn,
    FastSeparatePatchDataset, fast_separate_patch_collate_fn)
# from rectified.rectified_train_ddp import generate_images_with_rectified_flow
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match
from rectified.utils import generate_images_with_rectified_flow
from rectified.utils_uni2h import (
    load_uni2_h_model, extract_uni2_h_embeddings, extended_biological_evaluation_uni2h,
    calculate_uni2h_fid
)
from rectified.utils_he2rna import *
from rectified.utils_plot import save_all_evaluation_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_ddp():
    """Initialize DDP environment"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    return rank, local_rank, world_size, device


def cleanup_ddp():
    """Cleanup DDP environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def gather_list_across_ranks(data_list, world_size):
    """Gather lists from all ranks"""
    if world_size == 1:
        return data_list
    
    # Convert to tensor for gathering
    if len(data_list) == 0:
        return []
    
    # Gather all data
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data_list)
    
    # Flatten the gathered data
    all_data = []
    for rank_data in gathered_data:
        all_data.extend(rank_data)
    
    return all_data

class InceptionModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = torch.nn.Identity()
        for param in self.inception_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]

        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        x = (x * 2) - 1
        x = self.inception_model(x)
        return x


def calculate_fid(real_features, gen_features):
    if real_features.shape[0] < 2 or gen_features.shape[0] < 2:
        return np.nan

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    if np.isnan(sigma1).any() or np.isnan(sigma2).any() or np.isinf(sigma1).any() or np.isinf(sigma2).any():
        return np.nan

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return np.nan
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid if fid >= 0 and not np.isnan(fid) and not np.isinf(fid) else np.nan


def calculate_image_metrics(real_images_batch, generated_images_batch):
    batch_size = real_images_batch.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        real_img_np = real_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        gen_img_np = generated_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        
        real_img_np = np.clip(real_img_np, 0, 1)
        gen_img_np = np.clip(gen_img_np, 0, 1)

        real_img_rgb = real_img_np[:,:,:3]
        gen_img_rgb = gen_img_np[:,:,:3]

        data_range = 1.0 

        current_ssim = ssim(
            real_img_rgb, 
            gen_img_rgb, 
            channel_axis=2,
            data_range=data_range
        )
        
        current_psnr = psnr(
            real_img_rgb, 
            gen_img_rgb, 
            data_range=data_range
        )
        
        ssim_scores.append(current_ssim)
        psnr_scores.append(current_psnr)
    
    return ssim_scores, psnr_scores


def load_checkpoint_ddp(filename, model, device):
    """Load checkpoint and handle DDP model state dict"""
    checkpoint = torch.load(filename, map_location=device)
    
    model_state_dict = checkpoint.get("model", checkpoint)
    
    # Handle DDP model state dict (remove 'module.' prefix if present)
    if isinstance(model, DDP):
        # If loading into DDP model, ensure state dict matches
        if not any(key.startswith('module.') for key in model_state_dict.keys()):
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
    else:
        # If loading into non-DDP model, remove 'module.' prefix
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    return checkpoint

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RNA to H&E model with Rectified Flow (DDP-enabled)."
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model checkpoint (.pt or .pth).')
    parser.add_argument('--batch_size', type=int, default=20, help='Per-GPU batch size for evaluation.')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum number of samples to use from the dataset.')
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel evaluation.')
    parser.add_argument('--he2rna_model_path', type=str, default='/depot/natallah/data/Mengbo/HnE_RNA/GeneFlow/sequoia/models/he2rna-skcm-0', 
                           help='Path to pretrained HE2RNA model for RNA prediction validation')
    parser.add_argument('--save_embeddings', action='store_true', default=True, help='Save UNI2-h embeddings for later analysis')
    parser.add_argument('--embeddings_output_path', type=str, default=None, help='Path to save embeddings (if None, saves to output_dir/embeddings)')
    
    parser = setup_parser(parser)
    args = parser.parse_args()

    # Initialize DDP if requested
    if args.use_ddp:
        rank, local_rank, world_size, device = setup_ddp()
        # Adjust batch size for DDP
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
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Load UNI2-h model for biological validation and FID calculation
    uni2h_model, uni2h_processor, uni2h_preprocess = load_uni2_h_model(device)
    if rank == 0:
        if uni2h_model is not None:  # ← Check model, not processor
            logger.info("UNI2-h model loaded successfully for biological validation")
        else:
            logger.info("UNI2-h model unavailable, using ResNet fallback")

    # Load HE2RNA model for RNA prediction validation
    he2rna_model = None
    if args.he2rna_model_path and rank == 0:
        he2rna_model = load_he2rna_model(args.he2rna_model_path, device)
        if he2rna_model is not None:
            logger.info("HE2RNA model loaded successfully for RNA prediction validation")
        else:
            logger.info("HE2RNA model unavailable, skipping RNA prediction validation")

    # Data loading logic (same as original but with distributed considerations)
    expr_df = None
    gene_names = None
    missing_gene_symbols_list = None

    if args.hest1k_base_dir:
        if args.hest1k_sid is None or len(args.hest1k_sid) == 0:
            hest_metadata = pd.read_csv("/depot/natallah/data/Mengbo/HnE_RNA/data/HEST-1k/data/HEST_v1_1_0.csv")
            args.hest1k_sid = hest_metadata[(hest_metadata['st_technology']=='Xenium') &
                                            (hest_metadata['species']=='Homo sapiens')]['id'].tolist()
        if rank == 0:
            logger.info(f"Loading pre-processed HEST-1k data for sample {args.hest1k_sid}")
        expr_df, image_paths_dict = load_preprocessed_hest1k_singlecell_data(
            args.hest1k_sid, args.hest1k_base_dir, img_size=args.img_size, img_channels=args.img_channels)
        missing_gene_symbols_list = None
    elif args.hest1k_xenium_dir or args.hest1k_xenium_fast_dir:
        if rank == 0:
            logger.info(f"Preparing to load manually processed HEST-1k Xenium samples")
    elif args.adata is not None:
        if rank == 0:
            logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        if rank == 0:
            logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
        missing_gene_symbols_list = None

    if expr_df is not None and rank == 0:
        logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
        gene_names = expr_df.columns.tolist()

    # Dataset creation logic (same as training script)
    if args.model_type == 'single':
        if rank == 0:
            logger.info("Creating single-cell dataset for evaluation")
        if not (args.hest1k_sid and args.hest1k_base_dir):
            image_paths_dict = {}
            if args.image_paths:
                if rank == 0:
                    logger.info(f"Loading image paths from {args.image_paths}")
                with open(args.image_paths, "r") as f:
                    image_paths_data = json.load(f)
                image_paths_dict = {k: v for k, v in image_paths_data.items() if os.path.exists(v)}
                if rank == 0:
                    logger.info(f"Loaded {len(image_paths_dict)} valid cell image paths")

        full_dataset = CellImageGeneDataset(
            expr_df, image_paths_dict, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols_list,
            normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell
        if rank == 0:
            logger.info("Creating multi-cell dataset for evaluation")
        if args.hest1k_xenium_fast_dir is not None:
            if rank == 0:
                logger.info(f"Loading fast reformatted HEST-1k Xenium dataset from {args.hest1k_xenium_fast_dir}")
            full_dataset = FastSeparatePatchDataset(
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
            gene_names = full_dataset.gene_names
            if rank == 0:
                logger.info(f"Fast dataset loaded: {len(full_dataset)} patches, {len(gene_names)} unified genes")
        elif args.hest1k_xenium_dir is not None:
            if rank == 0:
                logger.info(f"Loading HEST-1k Xenium on-demand dataset from {args.hest1k_xenium_dir}")
            full_dataset = OnDemandMultiSampleHestXeniumDataset(
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
                sample_stats = full_dataset.get_sample_stats()
                logger.info("Sample statistics:")
                for sample_id, stats in sample_stats.items():
                    logger.info(f"  {sample_id}: {stats['n_patches']} patches, "
                              f"{stats['n_cells']} cells, {stats['n_genes']} genes")
        else:
            patch_image_paths_dict = None
            with open(args.patch_cell_mapping, "r") as f:
                patch_to_cells = json.load(f)
            if args.patch_image_paths:
                if rank == 0:
                    logger.info(f"Loading patch image paths from {args.patch_image_paths}")
                with open(args.patch_image_paths, "r") as f:
                    patch_image_paths_data = json.load(f)
                patch_image_paths_dict = {k: v for k, v in patch_image_paths_data.items() if os.path.exists(v)}
                if rank == 0:
                    logger.info(f"Loaded {len(patch_image_paths_dict)} valid patch image paths")

            full_dataset = PatchImageGeneDataset(
                expr_df=expr_df, patch_image_paths=patch_image_paths_dict, patch_to_cells=patch_to_cells,
                img_size=args.img_size, img_channels=args.img_channels,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
                normalize_aux=args.normalize_aux,
            )

    if len(full_dataset) == 0:
        if rank == 0:
            logger.error("Full dataset is empty. Cannot proceed with evaluation.")
        if args.use_ddp:
            cleanup_ddp()
        return

    # Split into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)  # Consistent split across ranks
    )

    if len(eval_dataset) == 0:
        if rank == 0:
            logger.error("Evaluation dataset is empty after splitting. Exiting.")
        if args.use_ddp:
            cleanup_ddp()
        return

    if len(eval_dataset) > args.max_samples:
        logger.info(f"Limiting evaluation dataset to {args.max_samples} samples")
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(args.max_samples))

    # Create distributed sampler
    if args.use_ddp:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        )
    else:
        eval_sampler = None

    # Determine collate function based on model type and dataset type
    collate_fn_to_use = None
    if args.model_type == 'multi':
        if args.hest1k_xenium_fast_dir is not None:
            collate_fn_to_use = fast_separate_patch_collate_fn
            if rank == 0:
                logger.info("Using fast separate patch collate function for evaluation")
        elif args.hest1k_xenium_dir is not None:
            collate_fn_to_use = multi_sample_hest_xenium_collate_fn
            if rank == 0:
                logger.info("Using multi-sample Xenium collate function for evaluation")
        else:
            collate_fn_to_use = patch_collate_fn
            if rank == 0:
                logger.info("Using standard patch collate function for evaluation")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_to_use
    )

    if rank == 0:
        logger.info(f"Evaluation set size: {len(eval_dataset)}")

    # Model initialization - Fixed gene dimension handling
    if expr_df is not None:
        gene_dim = expr_df.shape[1]
    elif hasattr(full_dataset, 'gene_names'):
        gene_dim = len(full_dataset.gene_names)
        gene_names = full_dataset.gene_names
    else:
        gene_dim = len(gene_names)

    if gene_dim is None:
        raise ValueError("Could not determine gene dimension. Ensure dataset has gene_names attribute.")

    if rank == 0:
        logger.info(f"Gene dimension: {gene_dim}")

    # Model Loading with Config Consistency
    if rank == 0:
        logger.info(f"Loading pretrained model from {args.model_path}")
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        if rank == 0:
            logger.error(f"Model checkpoint not found at {args.model_path}")
        if args.use_ddp:
            cleanup_ddp()
        return

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        if rank == 0:
            logger.error(f"Checkpoint missing model weights. Available keys: {checkpoint.keys()}")
        if args.use_ddp:
            cleanup_ddp()
        return
    
    model_config_ckpt = checkpoint.get("config", {})

    # Prioritize config from checkpoint for critical architectural params
    ckpt_img_channels = model_config_ckpt.get('img_channels', args.img_channels)
    ckpt_img_size = model_config_ckpt.get('img_size', args.img_size)
    ckpt_model_type = model_config_ckpt.get('model_type', args.model_type)

    # Update args if you want them to reflect checkpoint's settings for model init
    args.img_channels = ckpt_img_channels
    args.img_size = ckpt_img_size

    use_gene_attention = model_config_ckpt.get('use_gene_attention', args.use_gene_attention)
    use_multi_head_attention = model_config_ckpt.get('use_multi_head_attention', args.use_multi_head_attention)
    use_feature_gating = model_config_ckpt.get('use_feature_gating', args.use_feature_gating)
    use_residual_blocks = model_config_ckpt.get('use_residual_blocks', args.use_residual_blocks)
    use_layer_norm = model_config_ckpt.get('use_layer_norm', args.use_layer_norm)
    use_gene_relations = model_config_ckpt.get('use_gene_relations', args.use_gene_relations)

    model_constructor_args = dict(
        rna_dim=gene_dim, img_channels=args.img_channels, img_size=args.img_size,
        model_channels=128, num_res_blocks=2, attention_resolutions=(16,),
        dropout=0.1, channel_mult=(1, 2, 2, 2), use_checkpoint=False,
        num_heads=2, num_head_channels=16, use_scale_shift_norm=True,
        resblock_updown=True, use_new_attention_order=True,
        concat_mask=model_config_ckpt.get('concat_mask', getattr(args, 'concat_mask', False)),
        relation_rank=model_config_ckpt.get('relation_rank', getattr(args, 'relation_rank', 50)),
    )

    model = None
    try:
        if ckpt_model_type == 'single':
            model = RNAtoHnEModel(**model_constructor_args)
        else:  # multi-cell
            model_constructor_args['num_aggregation_heads'] = model_config_ckpt.get('num_aggregation_heads', getattr(args, 'num_aggregation_heads', 4))
            model = MultiCellRNAtoHnEModel(**model_constructor_args)
        model.load_state_dict(model_state)
    except Exception as e:
        if rank == 0:
            logger.warning(f"Failed to load model with current constructor: {e}")
            logger.info("Attempting to load with feature detection from checkpoint...")
        
        # Inspect checkpoint to determine which features it has
        has_gene_attention = any('gene_attention' in k for k in model_state.keys())
        has_gene_relation = any('gene_relation_net' in k for k in model_state.keys())
        has_multi_head = any('multi_head' in k for k in model_state.keys())
        has_feature_gating = any('gate' in k.lower() for k in model_state.keys())
        
        if rank == 0:
            logger.info(f"Checkpoint features detected:")
            logger.info(f"  - Gene attention: {has_gene_attention}")
            logger.info(f"  - Gene relations: {has_gene_relation}")
            logger.info(f"  - Multi-head attention: {has_multi_head}")
            logger.info(f"  - Feature gating: {has_feature_gating}")
        
        # Build minimal constructor args based on what checkpoint has
        minimal_constructor_args = {
            'rna_dim': gene_dim,
            'img_channels': args.img_channels,
            'img_size': args.img_size,
            'model_channels': 128,
            'num_res_blocks': 2,
            'attention_resolutions': (16,),
            'dropout': 0.1,
            'channel_mult': (1, 2, 2, 2),
            'use_checkpoint': False,
            'num_heads': 2,
            'num_head_channels': 16,
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            'use_new_attention_order': True,
        }
        
        # Try different model constructors in order of decreasing complexity
        model = None
        constructors_to_try = []
        
        if ckpt_model_type == 'single':
            # Try current model first (if checkpoint has all features)
            if has_gene_attention and has_gene_relation:
                constructors_to_try.append(('current', RNAtoHnEModel, {
                    **minimal_constructor_args,
                    'concat_mask': False,
                    'relation_rank': 50,
                    'use_gene_attention': True,
                    'use_multi_head_attention': has_multi_head,
                    'use_feature_gating': has_feature_gating,
                    'use_residual_blocks': True,
                    'use_layer_norm': True,
                    'use_gene_relations': True,
                }))
            # Try deprecated model
            constructors_to_try.append(('deprecated', RNAtoHnEModel_deprecation, minimal_constructor_args))
        else:  # multi-cell
            # Try current model with detected features
            if has_gene_relation:
                constructors_to_try.append(('current_with_relations', MultiCellRNAtoHnEModel, {
                    **minimal_constructor_args,
                    'num_aggregation_heads': 4,
                    'concat_mask': False,
                    'relation_rank': 50,
                    'use_multi_head_attention': has_multi_head,
                    'use_feature_gating': has_feature_gating,
                    'use_residual_blocks': True,
                    'use_layer_norm': True,
                    'use_gene_relations': True,
                }))
            # Try current model without relations
            constructors_to_try.append(('current_minimal', MultiCellRNAtoHnEModel, {
                **minimal_constructor_args,
                'num_aggregation_heads': 4,
                'concat_mask': False,
                'relation_rank': 50,
                'use_multi_head_attention': False,
                'use_feature_gating': False,
                'use_residual_blocks': True,
                'use_layer_norm': True,
                'use_gene_relations': False,
            }))
            # Try deprecated model
            constructors_to_try.append(('deprecated', MultiCellRNAtoHnEModel_deprecation, {
                **minimal_constructor_args,
                'num_aggregation_heads': 4,
            }))
        
        # Try each constructor until one works
        for constructor_name, constructor_class, constructor_args in constructors_to_try:
            try:
                if rank == 0:
                    logger.info(f"Trying {constructor_name} constructor...")
                
                test_model = constructor_class(**constructor_args)
                
                # Handle DDP prefix
                test_state = model_state.copy()
                if any(key.startswith('module.') for key in test_state.keys()):
                    test_state = {k.replace('module.', ''): v for k, v in test_state.items()}
                
                # Try loading
                test_model.load_state_dict(test_state, strict=False)
                model = test_model
                
                if rank == 0:
                    logger.info(f"✓ Successfully loaded model with {constructor_name} constructor")
                break
                
            except Exception as load_error:
                if rank == 0:
                    logger.debug(f"✗ {constructor_name} failed: {load_error}")
                continue
        
        if model is None:
            if rank == 0:
                logger.error("Failed to load model with any available constructor")
                logger.error("Checkpoint may be incompatible or corrupted")
            if args.use_ddp:
                cleanup_ddp()
            return

    if rank == 0:
        logger.info(f"Model loaded successfully using {ckpt_model_type} constructor.")

    # Move model to device
    model.to(device)

    # Wrap model with DDP if using distributed evaluation
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")

    model.eval()

    # Initialize evaluation components
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    inception_model = InceptionModel(device)

    # Storage for metrics
    all_ssim_scores = []
    all_psnr_scores = []
    all_real_features_for_fid = []
    all_gen_features_for_fid = []
    per_sample_metrics_list = []
    all_batch_fids_list = []

    all_real_embeddings_with_metadata = []
    all_gen_embeddings_with_metadata = []

    # RNA prediction validation storage
    all_rna_prediction_metrics = []
    all_original_rna = []
    all_predicted_rna = []

    # UNI2-h FID and biological validation storage
    all_uni2h_fids_list = []
    all_biological_results = []
    all_real_uni2h_features_for_fid = []
    all_gen_uni2h_features_for_fid = []

    if rank == 0:
        logger.info(f"Starting evaluation on {len(eval_loader)} batches")

    with torch.no_grad():
        # Create progress bar only on rank 0
        if rank == 0:
            eval_pbar = tqdm(eval_loader, desc="Evaluating")
        else:
            eval_pbar = eval_loader

        for batch_idx, batch in enumerate(eval_pbar):
            sample_ids_in_batch = []
            real_images_tensor = batch['image'].to(device)
            current_batch_size = real_images_tensor.shape[0]

            if ckpt_model_type == 'single':
                gene_expr = batch['gene_expr'].to(device)
                sample_ids_in_batch = batch['cell_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)

                generated_images_tensor = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps,
                    gene_mask=gene_mask, is_multi_cell=False
                )
            else:  # multi-cell
                processed_batch = prepare_multicell_batch(batch, device)
                gene_expr = processed_batch['gene_expr']
                num_cells_info = processed_batch.get('num_cells')
                sample_ids_in_batch = batch['patch_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)

                generated_images_tensor = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps,
                    num_cells=num_cells_info, gene_mask=gene_mask, is_multi_cell=True
                )

            # Apply stain normalization if enabled
            if getattr(args, 'enable_stain_normalization', False) and args.img_channels >= 3:
                normalized_generated_images_list = []
                for j in range(current_batch_size):
                    real_img_np_j = real_images_tensor[j].cpu().numpy().transpose(1, 2, 0)
                    gen_img_np_j = generated_images_tensor[j].cpu().numpy().transpose(1, 2, 0)
                    real_rgb_j = real_img_np_j[:, :, :3]
                    gen_rgb_original_j = gen_img_np_j[:, :, :3]

                    if getattr(args, 'stain_normalization_method', '') == 'skimage_hist_match':
                        gen_rgb_normalized_j = normalize_staining_rgb_skimage_hist_match(
                            gen_rgb_original_j, real_rgb_j
                        )
                    else:
                        gen_rgb_normalized_j = gen_rgb_original_j

                    if args.img_channels > 3:
                        gen_aux_j = gen_img_np_j[:, :, 3:]
                        final_gen_img_np_j = np.concatenate((gen_rgb_normalized_j, gen_aux_j), axis=2)
                    else:
                        final_gen_img_np_j = gen_rgb_normalized_j

                    normalized_generated_images_list.append(
                        torch.from_numpy(final_gen_img_np_j.transpose(2,0,1)).to(device)
                    )
                generated_images_tensor = torch.stack(normalized_generated_images_list)

            if he2rna_model is not None and rank == 0:
                # Add debugging for the first batch only
                if batch_idx == 0:
                    logger.info("DEBUGGING FIRST BATCH RNA PREDICTION...")
                    real_rna_pred, gen_rna_pred, _ = debug_rna_prediction_comparison(
                        real_images_tensor, generated_images_tensor, he2rna_model, device
                    )
                else:
                    # Normal prediction for other batches
                    real_rna_pred, gen_rna_pred, rna_features = predict_and_compare_rna_from_images(
                        real_images_tensor, generated_images_tensor, he2rna_model, device
                    )
                
                if real_rna_pred is not None and gen_rna_pred is not None:
                    # Calculate RNA prediction comparison metrics (real vs generated images)
                    rna_metrics = calculate_rna_image_comparison_metrics(
                        real_rna_pred.cpu().numpy(), gen_rna_pred.cpu().numpy()
                    )
                    
                    # Log detailed metrics for first batch
                    if batch_idx == 0:
                        logger.info(f"First batch RNA metrics: {rna_metrics}")
                    
                    # Store for aggregation
                    all_rna_prediction_metrics.append(rna_metrics)
                    all_original_rna.append(real_rna_pred.cpu().numpy())  # Real image RNA predictions
                    all_predicted_rna.append(gen_rna_pred.cpu().numpy())  # Generated image RNA predictions

            # Calculate image metrics (SSIM, PSNR)
            ssim_scores, psnr_scores = calculate_image_metrics(real_images_tensor, generated_images_tensor)
            all_ssim_scores.extend(ssim_scores)
            all_psnr_scores.extend(psnr_scores)

            # Extract features for traditional FID calculation
            real_features = inception_model(real_images_tensor).cpu().numpy()
            gen_features = inception_model(generated_images_tensor).cpu().numpy()
            all_real_features_for_fid.append(real_features)
            all_gen_features_for_fid.append(gen_features)

            per_sample_feat_dists = []
            if real_features.shape[0] > 0: # Ensure batch is not empty
                for i in range(real_features.shape[0]):
                    r_feat = real_features[i]
                    g_feat = gen_features[i]
                    distance = np.linalg.norm(r_feat - g_feat)
                    per_sample_feat_dists.append(distance)

            # Calculate batch-wise traditional FID
            batch_fid = calculate_fid(real_features, gen_features)
            all_batch_fids_list.append(batch_fid)

            # UNI2-h FID calculation
            uni2h_fid = calculate_uni2h_fid(
                real_images_tensor, generated_images_tensor,
                uni2h_model, uni2h_processor, uni2h_preprocess, device
            )
            all_uni2h_fids_list.append(uni2h_fid)

            # Extract and store UNI2-h features for overall FID calculation
            real_uni2h_features = extract_uni2_h_embeddings(
                real_images_tensor, uni2h_model, uni2h_processor, uni2h_preprocess, device
            )
            gen_uni2h_features = extract_uni2_h_embeddings(
                generated_images_tensor, uni2h_model, uni2h_processor, uni2h_preprocess, device
            )
            
            # Store embeddings with metadata for saving
            if args.save_embeddings:
                for i in range(current_batch_size):
                    # Real image embeddings
                    real_embedding_entry = {
                        'sample_id': sample_ids_in_batch[i],
                        'embedding': real_uni2h_features[i],
                        'type': 'real',
                        'batch_idx': batch_idx,
                        'rank': rank
                    }
                    all_real_embeddings_with_metadata.append(real_embedding_entry)
                    
                    # Generated image embeddings
                    gen_embedding_entry = {
                        'sample_id': sample_ids_in_batch[i],
                        'embedding': gen_uni2h_features[i],
                        'type': 'generated',
                        'batch_idx': batch_idx,
                        'rank': rank
                    }
                    all_gen_embeddings_with_metadata.append(gen_embedding_entry)

            all_real_uni2h_features_for_fid.append(real_uni2h_features)
            all_gen_uni2h_features_for_fid.append(gen_uni2h_features)

            # Batch-level biological validation
            biological_results = extended_biological_evaluation_uni2h(
                real_images_tensor, generated_images_tensor, 
                uni2h_model, uni2h_processor, uni2h_preprocess, device
            )

            # Store as batch-level metrics
            batch_biological_metrics = {
                'batch_idx': batch_idx,
                'rank': rank,
                'batch_size': current_batch_size,
                'uni2h_fid': uni2h_fid,
                'inception_fid': batch_fid
            }
            batch_biological_metrics.update(biological_results)
            all_biological_results.append(batch_biological_metrics)

            # Store per-sample metrics
            for i in range(current_batch_size):
                sample_metrics = {
                    'sample_id': sample_ids_in_batch[i],
                    'ssim': ssim_scores[i],
                    'psnr': psnr_scores[i],
                    'batch_idx': batch_idx,
                    'inception_feature_distance': per_sample_feat_dists[i] if i < len(per_sample_feat_dists) else np.nan,
                    'rank': rank
                }
                per_sample_metrics_list.append(sample_metrics)

            # Update progress bar (only on rank 0)
            if rank == 0:
                postfix_dict = {
                    'Avg SSIM': f'{np.mean(all_ssim_scores):.4f}',
                    'Avg PSNR': f'{np.mean(all_psnr_scores):.4f}',
                    'Inception FID': f'{batch_fid:.2f}' if not np.isnan(batch_fid) else 'N/A',
                    'UNI2-h FID': f'{uni2h_fid:.2f}' if not np.isnan(uni2h_fid) else 'N/A'
                }

                # Add RNA correlation if available
                if all_rna_prediction_metrics:
                    latest_rna_corr = all_rna_prediction_metrics[-1].get('rna_image_gene_correlation_mean', 0)
                    postfix_dict['RNA Corr'] = f'{latest_rna_corr:.4f}'
                
                eval_pbar.set_postfix(postfix_dict)

    # Gather results from all ranks
    if args.use_ddp:
        # Synchronize before gathering results
        dist.barrier()
        
        # Gather metrics from all ranks
        all_ssim_scores = gather_list_across_ranks(all_ssim_scores, world_size)
        all_psnr_scores = gather_list_across_ranks(all_psnr_scores, world_size)
        per_sample_metrics_list = gather_list_across_ranks(per_sample_metrics_list, world_size)
        all_batch_fids_list = gather_list_across_ranks(all_batch_fids_list, world_size)
        
        # Gather UNI2-h results
        all_uni2h_fids_list = gather_list_across_ranks(all_uni2h_fids_list, world_size)
        all_biological_results = gather_list_across_ranks(all_biological_results, world_size)

        # Gather features for global FID calculation
        gathered_real_features = [None for _ in range(world_size)]
        gathered_gen_features = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_real_features, all_real_features_for_fid)
        dist.all_gather_object(gathered_gen_features, all_gen_features_for_fid)

        # Gather UNI2-h features for global FID calculation
        gathered_real_uni2h_features = [None for _ in range(world_size)]
        gathered_gen_uni2h_features = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_real_uni2h_features, all_real_uni2h_features_for_fid)
        dist.all_gather_object(gathered_gen_uni2h_features, all_gen_uni2h_features_for_fid)
        
        # Gather embeddings with metadata if saving
        if args.save_embeddings:
            all_real_embeddings_with_metadata = gather_list_across_ranks(all_real_embeddings_with_metadata, world_size)
            all_gen_embeddings_with_metadata = gather_list_across_ranks(all_gen_embeddings_with_metadata, world_size)
           
        # Handle RNA prediction results (only rank 0 has this data)
        if rank == 0 and all_rna_prediction_metrics:
            # RNA data only exists on rank 0, no need to gather across ranks
            pass  # Keep the data as is on rank 0
        else:
            # Other ranks have no RNA data, initialize empty lists
            all_rna_prediction_metrics = []
            all_original_rna = []
            all_predicted_rna = []

        # Flatten features on rank 0
        if rank == 0:
            all_real_features_for_fid = []
            all_gen_features_for_fid = []
            for rank_real_features in gathered_real_features:
                all_real_features_for_fid.extend(rank_real_features)
            for rank_gen_features in gathered_gen_features:
                all_gen_features_for_fid.extend(rank_gen_features)

            # Flatten UNI2-h features on rank 0
            all_real_uni2h_features_for_fid = []
            all_gen_uni2h_features_for_fid = []
            for rank_real_features in gathered_real_uni2h_features:
                all_real_uni2h_features_for_fid.extend(rank_real_features)
            for rank_gen_features in gathered_gen_uni2h_features:
                all_gen_uni2h_features_for_fid.extend(rank_gen_features)

    # Compute final metrics and save results (only on rank 0)
    if rank == 0:
        logger.info("Computing final evaluation metrics...")
        
        # Calculate overall traditional FID
        all_real_features_concat = np.concatenate(all_real_features_for_fid, axis=0)
        all_gen_features_concat = np.concatenate(all_gen_features_for_fid, axis=0)
        overall_fid = calculate_fid(all_real_features_concat, all_gen_features_concat)

        # Calculate overall UNI2-h FID
        try:
            all_real_uni2h_features_concat = np.concatenate(all_real_uni2h_features_for_fid, axis=0)
            all_gen_uni2h_features_concat = np.concatenate(all_gen_uni2h_features_for_fid, axis=0)
            overall_uni2h_fid = calculate_fid(all_real_uni2h_features_concat, all_gen_uni2h_features_concat)
            logger.info(f"Overall UNI2-h FID calculated from {all_real_uni2h_features_concat.shape[0]} samples")
        except Exception as e:
            logger.warning(f"Could not compute overall UNI2-h FID: {e}")
            overall_uni2h_fid = np.nan
        
        # Calculate mean metrics
        mean_ssim = np.mean(all_ssim_scores)
        mean_psnr = np.mean(all_psnr_scores)
        std_ssim = np.std(all_ssim_scores)
        std_psnr = np.std(all_psnr_scores)

        all_feat_dists = [item['inception_feature_distance'] for item in per_sample_metrics_list if not np.isnan(item.get('inception_feature_distance', np.nan))]
        feat_dist_mean, feat_dist_std = (np.mean(all_feat_dists), np.std(all_feat_dists)) if all_feat_dists else (np.nan, np.nan)

        # Filter out NaN FID values
        valid_fids = [fid for fid in all_batch_fids_list if not np.isnan(fid)]
        mean_batch_fid = np.mean(valid_fids) if valid_fids else np.nan
        
        # Compute UNI2-h FID summary
        valid_uni2h_fids = [fid for fid in all_uni2h_fids_list if not np.isnan(fid)]
        mean_uni2h_fid = np.mean(valid_uni2h_fids) if valid_uni2h_fids else np.nan

        # Compute biological validation summary from batch-level results
        biological_summary = {}
        if all_biological_results:
            biological_keys = [k for k in all_biological_results[0].keys() 
                            if k not in ['batch_idx', 'rank', 'batch_size', 'uni2h_fid', 'inception_fid']]
            
            for key in biological_keys:
                values = [r[key] for r in all_biological_results if r[key] is not None and not np.isnan(r[key])]
                if values:
                    biological_summary[f'mean_{key}'] = float(np.mean(values))
                    biological_summary[f'std_{key}'] = float(np.std(values))

        # Compute RNA prediction summary
        rna_summary = {}
        if all_rna_prediction_metrics:
            for key in all_rna_prediction_metrics[0].keys():
                values = [m[key] for m in all_rna_prediction_metrics if key in m and not np.isnan(m[key])]
                if values:
                    rna_summary[f'mean_{key}'] = float(np.mean(values))
                    rna_summary[f'std_{key}'] = float(np.std(values))

        # results summary
        results_summary = {
            'total_samples': len(all_ssim_scores),
            'mean_ssim': float(mean_ssim),
            'std_ssim': float(std_ssim),
            'mean_psnr': float(mean_psnr),
            'std_psnr': float(std_psnr),
            'overall_fid': float(overall_fid),
            'mean_batch_fid': float(mean_batch_fid),
            'overall_uni2h_fid': float(overall_uni2h_fid),
            'mean_uni2h_fid': float(mean_uni2h_fid),
            'model_path': args.model_path,
            'model_type': ckpt_model_type,
            'img_size': args.img_size,
            'img_channels': args.img_channels,
            'generation_steps': args.gen_steps,
            'batch_size': original_batch_size,
            'world_size': world_size,
            'inception_feature_distance_mean': float(feat_dist_mean) if not np.isnan(feat_dist_mean) else None,
            'inception_feature_distance_std': float(feat_dist_std) if not np.isnan(feat_dist_std) else None,
            'use_gene_attention': use_gene_attention,
            'use_multi_head_attention': use_multi_head_attention,
            'use_feature_gating': use_feature_gating,
            'use_residual_blocks': use_residual_blocks,
            'use_layer_norm': use_layer_norm,
            'use_gene_relations': use_gene_relations,
        }
        
        # Add biological metrics to results summary
        results_summary.update(biological_summary)
        results_summary.update(rna_summary)

        # Save UNI2-h embeddings for UMAP analysis
        if args.save_embeddings:
            logger.info("Saving UNI2-h embeddings for UMAP analysis...")
            
            embeddings_dir = args.embeddings_output_path if args.embeddings_output_path else os.path.join(args.output_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Combine real and generated embeddings
            all_embeddings_with_metadata = all_real_embeddings_with_metadata + all_gen_embeddings_with_metadata
            
            # Create arrays for embeddings and metadata
            embeddings = np.array([entry['embedding'] for entry in all_embeddings_with_metadata])
            sample_ids = [entry['sample_id'] for entry in all_embeddings_with_metadata]
            types = [entry['type'] for entry in all_embeddings_with_metadata]
            batch_indices = [entry['batch_idx'] for entry in all_embeddings_with_metadata]
            
            # Save embeddings
            np.save(os.path.join(embeddings_dir, 'uni2h_embeddings.npy'), embeddings)
            
            # Save metadata
            metadata_df = pd.DataFrame({
                'sample_id': sample_ids,
                'type': types,
                'batch_idx': batch_indices,
                'embedding_index': range(len(sample_ids))
            })
            metadata_df.to_csv(os.path.join(embeddings_dir, 'embeddings_metadata.csv'), index=False)
            
            # Also save separate arrays for convenience
            real_embeddings = np.array([entry['embedding'] for entry in all_real_embeddings_with_metadata])
            gen_embeddings = np.array([entry['embedding'] for entry in all_gen_embeddings_with_metadata])
            
            np.save(os.path.join(embeddings_dir, 'uni2h_real_embeddings.npy'), real_embeddings)
            np.save(os.path.join(embeddings_dir, 'uni2h_generated_embeddings.npy'), gen_embeddings)
            
            # Save sample IDs separately
            real_sample_ids = [entry['sample_id'] for entry in all_real_embeddings_with_metadata]
            gen_sample_ids = [entry['sample_id'] for entry in all_gen_embeddings_with_metadata]
            
            with open(os.path.join(embeddings_dir, 'real_sample_ids.json'), 'w') as f:
                json.dump(real_sample_ids, f)
            with open(os.path.join(embeddings_dir, 'generated_sample_ids.json'), 'w') as f:
                json.dump(gen_sample_ids, f)
            
            logger.info(f"Saved {embeddings.shape[0]} UNI2-h embeddings ({embeddings.shape[1]} dimensions) to {embeddings_dir}")
            logger.info(f"Real embeddings: {real_embeddings.shape[0]}, Generated embeddings: {gen_embeddings.shape[0]}")
        
        # Save RNA predictions for detailed analysis
        if all_original_rna and all_predicted_rna:
            logger.info("Saving RNA prediction results...")
            np.save(os.path.join(args.output_dir, 'original_rna_expressions.npy'), 
                   np.concatenate(all_original_rna, axis=0))
            np.save(os.path.join(args.output_dir, 'predicted_rna_expressions.npy'), 
                   np.concatenate(all_predicted_rna, axis=0))
            
            # Save gene names if available
            if gene_names:
                with open(os.path.join(args.output_dir, 'gene_names.json'), 'w') as f:
                    json.dump(gene_names, f)
            
            logger.info(f"Saved RNA expressions: {len(all_original_rna)} batches")

        # logging with UNI2-h and biological validation results
        logger.info("="*80)
        logger.info("ENHANCED EVALUATION RESULTS WITH UNI2-H BIOLOGICAL VALIDATION")
        logger.info("(Biological metrics computed at batch level)")
        logger.info("="*80)
        logger.info(f"Total samples evaluated: {len(all_ssim_scores)}")
        logger.info(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        logger.info(f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}")
        logger.info(f"Inception FID: {overall_fid:.4f}")
        logger.info(f"UNI2-H FID (batch-wise mean): {mean_uni2h_fid:.4f}")
        logger.info(f"UNI2-H FID (overall): {overall_uni2h_fid:.4f}")
        logger.info("-" * 40)
        logger.info("BIOLOGICAL VALIDATION RESULTS:")
        logger.info("-" * 40)
        if biological_summary:
            logger.info(f"Cell Type Classification Accuracy: {biological_summary.get('mean_cell_type_accuracy', 0):.4f}")
            logger.info(f"UNI2-H Embedding Similarity: {biological_summary.get('mean_uni2h_embedding_similarity', 0):.4f}")
            logger.info(f"Nuclear Feature Similarity (avg): {np.mean([biological_summary.get(f'mean_nuclear_{f}_similarity', 0) for f in ['area', 'circularity', 'eccentricity', 'solidity']]):.4f}")
            logger.info(f"Spatial Pattern Similarity (avg): {np.mean([biological_summary.get(f'mean_spatial_{f}_similarity', 0) for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']]):.4f}")
            logger.info(f"Overall Biological Plausibility: {biological_summary.get('mean_overall_biological_plausibility', 0):.4f}")
        logger.info("="*80)
        logger.info("RNA PREDICTION VALIDATION (Round-trip: RNA→Image→RNA):")
        logger.info("-" * 40)
        if rna_summary:
            logger.info(f"Gene-wise RNA Correlation: {rna_summary.get('mean_rna_image_gene_correlation_mean', 0):.4f} ± {rna_summary.get('std_rna_image_gene_correlation_mean', 0):.4f}")
            logger.info(f"Sample-wise RNA Correlation: {rna_summary.get('mean_rna_image_sample_correlation_mean', 0):.4f} ± {rna_summary.get('std_rna_image_sample_correlation_mean', 0):.4f}")
            logger.info(f"Overall RNA Correlation: {rna_summary.get('mean_rna_image_overall_correlation', 0):.4f} ± {rna_summary.get('std_rna_image_overall_correlation', 0):.4f}")
            logger.info(f"RNA MSE: {rna_summary.get('mean_rna_image_mse_mean', 0):.4f} ± {rna_summary.get('std_rna_image_mse_mean', 0):.4f}")
            logger.info(f"Valid Gene Correlations: {rna_summary.get('mean_num_valid_gene_correlations', 0):.0f}")
            logger.info(f"Valid Sample Correlations: {rna_summary.get('mean_num_valid_sample_correlations', 0):.0f}")
            logger.info(f"Genes Compared: {rna_summary.get('mean_genes_compared', 0):.0f}")
        else:
            logger.info("No RNA prediction validation performed (HE2RNA model not provided)")
        logger.info("="*80)

        # Save summary to JSON
        with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)

        # Save per-sample metrics to CSV
        per_sample_df = pd.DataFrame(per_sample_metrics_list)
        per_sample_df.to_csv(os.path.join(args.output_dir, 'per_sample_metrics.csv'), index=False)

        # Save batch-level biological validation metrics separately
        biological_df = pd.DataFrame(all_biological_results)
        biological_df.to_csv(os.path.join(args.output_dir, 'batch_level_biological_validation.csv'), index=False)

        # Create and save all plots using the modular plotting functions
        plot_paths = save_all_evaluation_plots(
            args.output_dir,
            all_ssim_scores,
            all_psnr_scores,
            valid_fids,
            valid_uni2h_fids,
            all_biological_results,
            all_rna_prediction_metrics,
            results_summary,
            mean_ssim,
            mean_psnr,
            mean_batch_fid,
            mean_uni2h_fid,
            gene_names
        )
        
        logger.info(f"evaluation results saved to {args.output_dir}")

    # Cleanup DDP
    if args.use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()