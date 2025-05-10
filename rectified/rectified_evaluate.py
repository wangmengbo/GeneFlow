import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg

# For FID calculation
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata
from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation
from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn
from rectified.rectified_train import generate_images_with_rectified_flow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Inception model for FID calculation
class InceptionModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = torch.nn.Identity()
        for param in self.inception_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1: # Handle grayscale: repeat to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: # Handle >3 channels: use first 3
            x = x[:, :3, :, :]
        # Else, assume 3 channels

        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        x = (x * 2) - 1 # Assumes input images are [0, 1]
        x = self.inception_model(x)
        return x

# MODIFIED Calculate FID score
def calculate_fid(real_features, gen_features):
    # Need at least 2 samples to calculate covariance
    if real_features.shape[0] < 2 or gen_features.shape[0] < 2:
        # logger.debug("FID calculation requires at least 2 samples per set.")
        return np.nan

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Check for NaNs in covariance matrices (can happen if input features are constant)
    if np.isnan(sigma1).any() or np.isnan(sigma2).any():
        # logger.debug("NaNs in covariance matrix during FID calculation.")
        return np.nan

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    try:
        # disp=False to suppress warnings which can be verbose in a loop
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception as e:
        # logger.debug(f"Error in linalg.sqrtm during FID: {e}")
        return np.nan
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid if fid >= 0 else np.nan


# Calculate SSIM and PSNR for a batch of images (Original)
def calculate_image_metrics(real_images, generated_images):
    batch_size = real_images.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        real_img_rgb = real_img[:,:,:3]
        gen_img_rgb = gen_img[:,:,:3]

        ssim_score = ssim(
            real_img_rgb, 
            gen_img_rgb, 
            channel_axis=2, 
            data_range=1.0,
            multichannel=True 
        )
        
        psnr_score = psnr(
            real_img_rgb, 
            gen_img_rgb, 
            data_range=1.0
        )
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return ssim_scores, psnr_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA to H&E model with Rectified Flow using FID, SSIM, and PSNR metrics.")
    parser.add_argument('--model_path', type=str, default="cell_256_aux/output_rectified/best_single_rna_to_hne_model_rectified_multi-head.pt", help='Path to the pretrained model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_rectified', help='Directory to save outputs.')
    # <<< START MODIFICATION >>>
    parser.add_argument('--output_name_prefix', type=str, default='', help='Prefix for the output evaluation files.')
    # <<< END MODIFICATION >>>
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and evaluation.')
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
    parser.add_argument('--eval_samples', type=int, default=None, help='Number of samples to evaluate (None for all).')

    parser = setup_parser(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        logger.info(f"Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")

    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")
        if args.image_paths:
            logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f: image_paths_data = json.load(f) # Renamed to avoid conflict
            image_paths = {k: v for k, v in image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(image_paths)} valid cell image paths")
        else: image_paths = {}
        dataset = CellImageGeneDataset(
            expr_df, image_paths, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols if 'missing_gene_symbols' in locals() else None,
            normalize_aux=args.normalize_aux,
        )
    else:
        logger.info("Creating multi-cell dataset")
        with open(args.patch_cell_mapping, "r") as f: patch_to_cells = json.load(f)
        if args.patch_image_paths:
            logger.info(f"Loading patch image paths from {args.patch_image_paths}")
            with open(args.patch_image_paths, "r") as f: patch_image_paths_data = json.load(f) # Renamed
            patch_image_paths = {k: v for k, v in patch_image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(patch_image_paths)} valid patch image paths")
        else: patch_image_paths = {}
        dataset = PatchImageGeneDataset(
            expr_df=expr_df, patch_image_paths=patch_image_paths, patch_to_cells=patch_to_cells,
            img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            normalize_aux=args.normalize_aux,
        )
    logger.info(f"Dataset created with {len(dataset)} samples")

    if args.eval_samples is not None:
        eval_size = min(args.eval_samples, len(dataset))
        eval_indices = torch.randperm(len(dataset))[:eval_size].tolist()
    else:
        eval_size = min(int(0.2 * len(dataset)), len(dataset))
        if eval_size == 0 and len(dataset) > 0: eval_size = len(dataset)
        eval_indices = torch.randperm(len(dataset))[:eval_size].tolist()
    
    if not eval_indices and len(dataset) > 0 : # If 20% is 0, but dataset is not empty, take all
        logger.warning("Evaluation set was empty after sampling, using all dataset samples for evaluation.")
        eval_indices = list(range(len(dataset)))
    
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices) if eval_indices else []


    if not eval_dataset:
        logger.error("Evaluation dataset is empty. Exiting.")
        return

    collate_fn_to_use = patch_collate_fn if args.model_type == 'multi' else None
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_to_use
    )
    logger.info(f"Evaluation set size: {len(eval_dataset)}")

    gene_dim = expr_df.shape[1]
    args.relation_rank = getattr(args, 'relation_rank', 16) 
    args.concat_mask = getattr(args, 'concat_mask', False)

    model_constructor_args = dict(
        rna_dim=gene_dim, img_channels=args.img_channels, img_size=args.img_size,
        model_channels=128, num_res_blocks=2, attention_resolutions=(16,),
        dropout=0.1, channel_mult=(1, 2, 2, 2), use_checkpoint=False,
        num_heads=2, num_head_channels=16, use_scale_shift_norm=True,
        resblock_updown=True, use_new_attention_order=True, concat_mask=args.concat_mask,
    )

    logger.info(f"Loading pretrained model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model = None
    try:
        if args.model_type == 'single':
            model = RNAtoHnEModel(**model_constructor_args, relation_rank=args.relation_rank)
        else:
            model = MultiCellRNAtoHnEModel(**model_constructor_args, relation_rank=args.relation_rank)
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        logger.error(f"Failed to load model with current constructor: {e}")
        logger.warning(f"Attempting to load model with deprecated constructor.")
        if args.model_type == 'single':
            model = RNAtoHnEModel_deprecation(**model_constructor_args)
        else:
            model = MultiCellRNAtoHnEModel_deprecation(**model_constructor_args)
        model.load_state_dict(checkpoint["model"])
        
    logger.info(f"Model loaded successfully")
    model.to(device)
    model.eval()
    
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    inception_model = InceptionModel(device)
    
    all_ssim = []
    all_psnr = []
    all_real_features_for_fid = [] 
    all_gen_features_for_fid = []   
    per_sample_metrics_list = []
    all_batch_fids_list = [] # <-- ADDED: List to store valid per-batch FID scores

    logger.info(f"Starting evaluation on {len(eval_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            sample_ids_in_batch = []
            real_images = batch['image'].to(device)

            if args.model_type == 'single':
                gene_expr = batch['gene_expr'].to(device)
                sample_ids_in_batch = batch['cell_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None: gene_mask = gene_mask.to(device)
                
                generated_images = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps, gene_mask, False
                )
            else: 
                processed_batch = prepare_multicell_batch(batch, device)
                gene_expr = processed_batch['gene_expr']
                num_cells = processed_batch.get('num_cells') # Use .get for safety
                sample_ids_in_batch = batch['patch_id']
                
                generated_images = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps, num_cells=num_cells, is_multi_cell=True
                )
            
            real_features_batch = inception_model(real_images)
            gen_features_batch = inception_model(generated_images) # InceptionModel now handles channel adjustment
            
            all_real_features_for_fid.append(real_features_batch.cpu().numpy())
            all_gen_features_for_fid.append(gen_features_batch.cpu().numpy())
            
            ssim_scores, psnr_scores = calculate_image_metrics(real_images, generated_images)
            all_ssim.extend(ssim_scores)
            all_psnr.extend(psnr_scores)

            per_sample_feat_dists = []
            for i in range(real_features_batch.shape[0]):
                r_feat = real_features_batch[i].cpu().numpy()
                g_feat = gen_features_batch[i].cpu().numpy()
                distance = np.linalg.norm(r_feat - g_feat)
                per_sample_feat_dists.append(distance)

            # <-- ADDED: Per-Batch FID Calculation -->
            fid_batch = np.nan
            # .cpu().numpy() done above for all_real/gen_features_for_fid, reuse if possible or re-evaluate for clarity.
            # For clarity, let's use the already CPU-Numpy converted versions for this batch.
            real_features_batch_np = real_features_batch.cpu().numpy() 
            gen_features_batch_np = gen_features_batch.cpu().numpy()

            # calculate_fid now handles batch size check and try-except internally
            fid_batch = calculate_fid(real_features_batch_np, gen_features_batch_np)
            
            if not np.isnan(fid_batch):
                all_batch_fids_list.append(fid_batch)
            else:
                logger.debug(f"FID for batch {batch_idx} is NaN (batch size: {real_features_batch_np.shape[0]}).")
            # <-- END ADDED SECTION -->

            for i in range(len(ssim_scores)):
                per_sample_metrics_list.append({
                    'sample_id': sample_ids_in_batch[i],
                    'ssim': ssim_scores[i],
                    'psnr': psnr_scores[i],
                    'inception_feature_distance': per_sample_feat_dists[i],
                    'batch_fid': fid_batch # <-- ADDED: Store batch_fid per sample
                })
    
    # Global FID
    global_fid_score = np.nan 
    if len(all_real_features_for_fid) > 0 and len(all_gen_features_for_fid) > 0:
        all_real_features_np = np.concatenate(all_real_features_for_fid, axis=0)
        all_gen_features_np = np.concatenate(all_gen_features_for_fid, axis=0)
        # calculate_fid handles internal checks for sample size now
        global_fid_score = calculate_fid(all_real_features_np, all_gen_features_np)
        if np.isnan(global_fid_score):
            logger.warning(f"Global FID calculation resulted in NaN (total samples: {all_real_features_np.shape[0]}). Check feature quality or sample count.")
    else:
        logger.warning("No features collected for global FID calculation.")

    ssim_mean, ssim_std = (np.mean(all_ssim), np.std(all_ssim)) if all_ssim else (np.nan, np.nan)
    psnr_mean, psnr_std = (np.mean(all_psnr), np.std(all_psnr)) if all_psnr else (np.nan, np.nan)
    
    all_feat_dists = [item['inception_feature_distance'] for item in per_sample_metrics_list if 'inception_feature_distance' in item]
    feat_dist_mean, feat_dist_std = (np.mean(all_feat_dists), np.std(all_feat_dists)) if all_feat_dists else (np.nan, np.nan)

    # <-- ADDED: Calculate mean/std for per-batch FID scores -->
    batch_fid_mean = np.mean(all_batch_fids_list) if all_batch_fids_list else np.nan
    batch_fid_std = np.std(all_batch_fids_list) if all_batch_fids_list else np.nan
    # <-- END ADDED SECTION -->

    metrics_summary = {
        'global_fid': float(global_fid_score) if not np.isnan(global_fid_score) else None, # Renamed for clarity
        'ssim_mean': float(ssim_mean) if not np.isnan(ssim_mean) else None,
        'ssim_std': float(ssim_std) if not np.isnan(ssim_std) else None,
        'psnr_mean': float(psnr_mean) if not np.isnan(psnr_mean) else None,
        'psnr_std': float(psnr_std) if not np.isnan(psnr_std) else None,
        'inception_feature_distance_mean': float(feat_dist_mean) if not np.isnan(feat_dist_mean) else None,
        'inception_feature_distance_std': float(feat_dist_std) if not np.isnan(feat_dist_std) else None,
        'batch_fid_mean': float(batch_fid_mean) if not np.isnan(batch_fid_mean) else None, # <-- ADDED
        'batch_fid_std': float(batch_fid_std) if not np.isnan(batch_fid_std) else None,     # <-- ADDED
        'num_valid_batch_fids': len(all_batch_fids_list), # <-- ADDED
        'num_samples': len(all_ssim) if all_ssim else 0
    }
    
    # <<< START MODIFICATION: Construct output filenames with prefix >>>
    prefix = args.output_name_prefix if args.output_name_prefix else ""
    if prefix and not prefix.endswith("_"): # Add underscore if prefix exists and doesn't have one
        prefix += "_"

    summary_filename = f"{prefix}rectified_metrics_summary.json"
    csv_filename = f"{prefix}rectified_per_sample_metrics.csv"
    plot_filename = f"{prefix}rectified_metric_distributions.png"
    
    with open(os.path.join(args.output_dir, summary_filename), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics summary saved to {os.path.join(args.output_dir, summary_filename)}")
    
    if per_sample_metrics_list:
        per_sample_df = pd.DataFrame(per_sample_metrics_list)
        csv_output_path = os.path.join(args.output_dir, csv_filename)
        per_sample_df.to_csv(csv_output_path, index=False)
        logger.info(f"Per-sample metrics (including batch_fid) saved to {csv_output_path}")
    else:
        logger.info("No per-sample metrics to save.")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if all_ssim:
        plt.hist(all_ssim, bins=20, alpha=0.7); plt.axvline(ssim_mean, color='r', ls='dashed', lw=2)
        plt.title(f'SSIM (Mean: {ssim_mean:.4f})')
    else: plt.title('SSIM Distribution (No data)')
    plt.xlabel('SSIM'); plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    if all_psnr:
        plt.hist(all_psnr, bins=20, alpha=0.7); plt.axvline(psnr_mean, color='r', ls='dashed', lw=2)
        plt.title(f'PSNR (Mean: {psnr_mean:.4f})')
    else: plt.title('PSNR Distribution (No data)')
    plt.xlabel('PSNR (dB)'); plt.ylabel('Count')
    
    plt.tight_layout()
    plot_output_path = os.path.join(args.output_dir, plot_filename)
    plt.savefig(plot_output_path)
    logger.info(f"Metric distributions plot saved to {plot_output_path}")
    # <<< END MODIFICATION >>>
    
    logger.info(f"=== Evaluation Results (Aggregated) ===")
    if metrics_summary['num_samples'] > 0 :
        logger.info(f"Number of samples: {metrics_summary['num_samples']}")
        logger.info(f"Global FID Score (Dataset Level): {metrics_summary['global_fid'] if metrics_summary['global_fid'] is not None else 'N/A'}")
        logger.info(f"SSIM: {metrics_summary['ssim_mean']:.4f} ± {metrics_summary['ssim_std']:.4f}" if metrics_summary['ssim_mean'] is not None else "SSIM: N/A")
        logger.info(f"PSNR: {metrics_summary['psnr_mean']:.4f} ± {metrics_summary['psnr_std']:.4f}" if metrics_summary['psnr_mean'] is not None else "PSNR: N/A")
        logger.info(f"Per-Sample Inception Feature Distance: {metrics_summary['inception_feature_distance_mean']:.4f} ± {metrics_summary['inception_feature_distance_std']:.4f}" if metrics_summary['inception_feature_distance_mean'] is not None else "Inception Feature Distance: N/A")
        # <-- ADDED: Log batch FID summary -->
        if metrics_summary['num_valid_batch_fids'] > 0:
            logger.info(f"Per-Batch FID (Mean over {metrics_summary['num_valid_batch_fids']} batches): {metrics_summary['batch_fid_mean']:.4f} ± {metrics_summary['batch_fid_std']:.4f}")
        else:
            logger.info("Per-Batch FID: N/A (no valid batches or all resulted in NaN)")
        # <-- END ADDED SECTION -->
    else:
        logger.info("No samples were evaluated.")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
