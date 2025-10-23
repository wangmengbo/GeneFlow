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

# Inception model for FID calculation
class InceptionModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device) # Use updated weights argument
        self.inception_model.eval()
        self.inception_model.fc = torch.nn.Identity()
        for param in self.inception_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Input x expected to be in [0, 1] range, B,C,H,W
        # Handle channel differences for InceptionV3 input
        if x.shape[1] == 1: # Handle grayscale: repeat to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: # Handle >3 channels: use first 3
            x = x[:, :3, :, :]
        # Else, assume 3 channels and x is B,3,H,W

        # Preprocessing for InceptionV3: normalize to [-1, 1] and resize
        # InceptionV3 expects 299x299 images
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] if input is [0, 1]
        x = (x * 2) - 1
        
        x = self.inception_model(x)
        return x

# Calculate FID score
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


# Calculate SSIM and PSNR for a batch of images
def calculate_image_metrics(real_images_batch, generated_images_batch):
    # Expects PyTorch Tensors (B, C, H, W) in range [0,1]
    batch_size = real_images_batch.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        # Convert to H, W, C numpy arrays
        real_img_np = real_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        gen_img_np = generated_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        
        # Ensure they are in [0,1] for metrics
        real_img_np = np.clip(real_img_np, 0, 1)
        gen_img_np = np.clip(gen_img_np, 0, 1)

        # Metrics are typically calculated on RGB
        real_img_rgb = real_img_np[:,:,:3]
        gen_img_rgb = gen_img_np[:,:,:3]

        # Data range for ssim/psnr for float images is typically 1.0 (max-min)
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA to H&E model with Diffusion using FID, SSIM, and PSNR metrics.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_diffusion', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels.')
    parser.add_argument('--gen_steps', type=int, default=300, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels during dataset loading.')
    parser.add_argument('--diffusion_timesteps', type=int, default=300, help='Number of timesteps for diffusion process')
    parser.add_argument('--beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Noise schedule for diffusion')
    parser.add_argument('--predict_noise', action='store_true', default=True, help='Whether model predicts noise (True) or x_0 (False)') # Default changed to True as common
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method for diffusion generation')
    
    # Arguments for stain normalization and others will be added by setup_parser
    parser = setup_parser(parser)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    missing_gene_symbols_list = None # Initialize
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        logger.info(f"Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
    
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset for evaluation")
        image_paths_dict = {}
        if args.image_paths:
            logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f: image_paths_data = json.load(f)
            image_paths_dict = {k: v for k, v in image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(image_paths_dict)} valid cell image paths")
        
        full_dataset = CellImageGeneDataset(
            expr_df, image_paths_dict, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols_list,
            normalize_aux=args.normalize_aux,
        )
    else: # multi-cell
        logger.info("Creating multi-cell dataset for evaluation")
        patch_image_paths_dict = None
        with open(args.patch_cell_mapping, "r") as f: patch_to_cells = json.load(f)
        if args.patch_image_paths:
            logger.info(f"Loading patch image paths from {args.patch_image_paths}")
            with open(args.patch_image_paths, "r") as f: patch_image_paths_data = json.load(f)
            patch_image_paths_dict = {k: v for k, v in patch_image_paths_data.items() if os.path.exists(v)}
            logger.info(f"Loaded {len(patch_image_paths_dict)} valid patch image paths")
        
        full_dataset = PatchImageGeneDataset(
            expr_df=expr_df, patch_image_paths=patch_image_paths_dict, patch_to_cells=patch_to_cells,
            img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            normalize_aux=args.normalize_aux,
        )
    
    if len(full_dataset) == 0:
        logger.error("Full dataset is empty. Cannot proceed with evaluation.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size == 0 and len(full_dataset) > 0: # Ensure val_set is not empty if full_dataset has samples
        if train_size > 0 : # if full_dataset has at least 2 samples
            train_size = len(full_dataset) -1
            val_size = 1
        else: # full_dataset has 1 sample, use it for validation
             eval_dataset = full_dataset
             logger.warning("Full dataset has only 1 sample. Using it entirely for evaluation.")
    
    if val_size > 0 :
         _, eval_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    elif not 'eval_dataset' in locals(): # handles the case where full_dataset was 0 len and eval_dataset not assigned.
        logger.error("Evaluation dataset could not be created due to insufficient samples in the full dataset.")
        return


    if len(eval_dataset) == 0:
        logger.error("Evaluation dataset is empty after splitting. Exiting.")
        return

    collate_fn_to_use = patch_collate_fn if args.model_type == 'multi' else None
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(os.cpu_count(), 4), pin_memory=True, collate_fn=collate_fn_to_use
    )
    logger.info(f"Evaluation set size: {len(eval_dataset)}")

    gene_dim = expr_df.shape[1]
    
    logger.info(f"Loading pretrained model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return
        
    model_state = checkpoint.get("model", checkpoint)
    model_config_ckpt = checkpoint.get("config", {})

    args.img_channels = model_config_ckpt.get('img_channels', args.img_channels)

    model_constructor_args = dict(
        rna_dim=gene_dim, 
        img_channels=args.img_channels, # Use potentially updated args.img_channels
        img_size=args.img_size,
        model_channels=model_config_ckpt.get('model_channels', 128), # Example: allow override from ckpt
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
        relation_rank= model_config_ckpt.get('relation_rank', getattr(args, 'relation_rank', 50))
    )
    
    model = None
    # Using args.model_type from command line for diffusion model selection
    if args.model_type == 'single':
        model = RNAtoHnEModel(**model_constructor_args)
    else: # multi-cell
        model_constructor_args['num_aggregation_heads'] = model_config_ckpt.get('num_aggregation_heads', getattr(args, 'num_aggregation_heads', 4))
        model = MultiCellRNAtoHnEModel(**model_constructor_args)
    
    model.load_state_dict(model_state)
    logger.info(f"Model loaded successfully using {args.model_type} constructor with img_channels={args.img_channels}.")
    model.to(device)
    model.eval()
    
    diffusion = GaussianDiffusion(
        timesteps=args.diffusion_timesteps,
        beta_schedule=args.beta_schedule,
        predict_noise=args.predict_noise,
        device=device
    )
    
    inception_model = InceptionModel(device)
    
    all_ssim_scores = []
    all_psnr_scores = []
    all_real_features_for_fid = [] 
    all_gen_features_for_fid = []   
    per_sample_metrics_list = []
    all_batch_fids_list = [] 

    logger.info(f"Starting evaluation on {len(eval_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            sample_ids_in_batch = []
            real_images_tensor = batch['image'].to(device) # B,C,H,W
            current_batch_size = real_images_tensor.shape[0]

            if args.model_type == 'single':
                gene_expr = batch['gene_expr'].to(device)
                sample_ids_in_batch = batch['cell_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None: gene_mask = gene_mask.to(device)
                
                generated_images_tensor = generate_images_with_diffusion(
                    model=model, diffusion=diffusion, gene_expr=gene_expr, device=device,
                    num_steps=args.gen_steps, gene_mask=gene_mask, is_multi_cell=False, method=args.sampling_method
                )
            else: # multi-cell
                processed_batch = prepare_multicell_batch(batch, device)
                gene_expr = processed_batch['gene_expr']
                num_cells_info = processed_batch.get('num_cells') 
                sample_ids_in_batch = batch['patch_id']
                gene_mask = batch.get('gene_mask', None) # Assuming gene_mask can also be part of multi-cell batch
                if gene_mask is not None: gene_mask = gene_mask.to(device)

                generated_images_tensor = generate_images_with_diffusion(
                    model=model, diffusion=diffusion, gene_expr=gene_expr, device=device,
                    num_steps=args.gen_steps, num_cells=num_cells_info, gene_mask=gene_mask, is_multi_cell=True, method=args.sampling_method
                )
            
            if args.enable_stain_normalization and args.img_channels >= 3:
                logger.debug(f"Applying stain normalization for batch {batch_idx} using method: {args.stain_normalization_method}")
                normalized_generated_images_list = []
                for j in range(current_batch_size):
                    real_img_np_j = real_images_tensor[j].cpu().numpy().transpose(1, 2, 0) # H,W,C
                    gen_img_np_j = generated_images_tensor[j].cpu().numpy().transpose(1, 2, 0) # H,W,C

                    real_rgb_j = real_img_np_j[:, :, :3]
                    gen_rgb_original_j = gen_img_np_j[:, :, :3]

                    if args.stain_normalization_method == 'skimage_hist_match':
                        gen_rgb_normalized_j = normalize_staining_rgb_skimage_hist_match(
                            gen_rgb_original_j, real_rgb_j
                        )
                    else:
                        # logger.warning(f"Unsupported stain norm method: {args.stain_normalization_method}. Using original.")
                        gen_rgb_normalized_j = gen_rgb_original_j
                    
                    if args.img_channels > 3: # If there are auxiliary channels
                        gen_aux_j = gen_img_np_j[:, :, 3:]
                        final_gen_img_np_j = np.concatenate((gen_rgb_normalized_j, gen_aux_j), axis=2)
                    else:
                        final_gen_img_np_j = gen_rgb_normalized_j
                    
                    # Convert back to C,H,W tensor and add to list
                    normalized_generated_images_list.append(
                        torch.from_numpy(final_gen_img_np_j.transpose(2,0,1)).to(device)
                    )
                generated_images_tensor = torch.stack(normalized_generated_images_list) # B,C,H,W

            # Ensure images are in [0,1] before passing to Inception or metrics
            real_images_tensor = torch.clamp(real_images_tensor, 0, 1)
            generated_images_tensor = torch.clamp(generated_images_tensor, 0, 1)

            real_features_batch = inception_model(real_images_tensor)
            gen_features_batch = inception_model(generated_images_tensor)
            
            all_real_features_for_fid.append(real_features_batch.cpu().numpy())
            all_gen_features_for_fid.append(gen_features_batch.cpu().numpy())
            
            batch_ssim_scores, batch_psnr_scores = calculate_image_metrics(real_images_tensor, generated_images_tensor)
            all_ssim_scores.extend(batch_ssim_scores)
            all_psnr_scores.extend(batch_psnr_scores)

            per_sample_feat_dists = []
            if real_features_batch.shape[0] > 0: # Ensure batch is not empty
                for i in range(real_features_batch.shape[0]):
                    r_feat = real_features_batch[i].cpu().numpy()
                    g_feat = gen_features_batch[i].cpu().numpy()
                    distance = np.linalg.norm(r_feat - g_feat)
                    per_sample_feat_dists.append(distance)

            fid_batch = calculate_fid(real_features_batch.cpu().numpy(), gen_features_batch.cpu().numpy())
            
            if not np.isnan(fid_batch):
                all_batch_fids_list.append(fid_batch)

            for i in range(len(batch_ssim_scores)):
                per_sample_metrics_list.append({
                    'sample_id': sample_ids_in_batch[i] if i < len(sample_ids_in_batch) else f"batch{batch_idx}_sample{i}",
                    'ssim': batch_ssim_scores[i],
                    'psnr': batch_psnr_scores[i],
                    'inception_feature_distance': per_sample_feat_dists[i] if i < len(per_sample_feat_dists) else np.nan,
                    'batch_fid': fid_batch 
                })
    
    global_fid_score = np.nan 
    if len(all_real_features_for_fid) > 0 and len(all_gen_features_for_fid) > 0:
        all_real_features_np = np.concatenate(all_real_features_for_fid, axis=0)
        all_gen_features_np = np.concatenate(all_gen_features_for_fid, axis=0)
        if all_real_features_np.shape[0] >=2 and all_gen_features_np.shape[0] >=2 :
            global_fid_score = calculate_fid(all_real_features_np, all_gen_features_np)
            if np.isnan(global_fid_score):
                logger.warning(f"Global FID calculation resulted in NaN (total samples: {all_real_features_np.shape[0]}). Check feature quality or sample count.")
        else:
            logger.warning(f"Not enough samples for global FID calculation (real: {all_real_features_np.shape[0]}, gen: {all_gen_features_np.shape[0]}). Needs at least 2.")
    else:
        logger.warning("No features collected for global FID calculation.")

    ssim_mean, ssim_std = (np.mean(all_ssim_scores), np.std(all_ssim_scores)) if all_ssim_scores else (np.nan, np.nan)
    psnr_mean, psnr_std = (np.mean(all_psnr_scores), np.std(all_psnr_scores)) if all_psnr_scores else (np.nan, np.nan)
    
    all_feat_dists = [item['inception_feature_distance'] for item in per_sample_metrics_list if not np.isnan(item.get('inception_feature_distance', np.nan))]
    feat_dist_mean, feat_dist_std = (np.mean(all_feat_dists), np.std(all_feat_dists)) if all_feat_dists else (np.nan, np.nan)

    batch_fid_mean = np.mean(all_batch_fids_list) if all_batch_fids_list else np.nan
    batch_fid_std = np.std(all_batch_fids_list) if all_batch_fids_list else np.nan

    metrics_summary = {
        'global_fid': float(global_fid_score) if not np.isnan(global_fid_score) else None,
        'ssim_mean': float(ssim_mean) if not np.isnan(ssim_mean) else None,
        'ssim_std': float(ssim_std) if not np.isnan(ssim_std) else None,
        'psnr_mean': float(psnr_mean) if not np.isnan(psnr_mean) else None,
        'psnr_std': float(psnr_std) if not np.isnan(psnr_std) else None,
        'inception_feature_distance_mean': float(feat_dist_mean) if not np.isnan(feat_dist_mean) else None,
        'inception_feature_distance_std': float(feat_dist_std) if not np.isnan(feat_dist_std) else None,
        'batch_fid_mean': float(batch_fid_mean) if not np.isnan(batch_fid_mean) else None, 
        'batch_fid_std': float(batch_fid_std) if not np.isnan(batch_fid_std) else None,    
        'num_valid_batch_fids': len(all_batch_fids_list), 
        'num_samples_evaluated': len(all_ssim_scores) if all_ssim_scores else 0,
        'stain_normalization_enabled': args.enable_stain_normalization, # Added
        'stain_normalization_method': args.stain_normalization_method if args.enable_stain_normalization else 'none', # Added
        'sampling_method': args.sampling_method
    }
    
    prefix = args.output_name_prefix if args.output_name_prefix else ""
    if prefix and not prefix.endswith("_"): 
        prefix += "_"

    summary_filename = f"{prefix}diffusion_metrics_summary.json"
    csv_filename = f"{prefix}diffusion_per_sample_metrics.csv"
    plot_filename = f"{prefix}diffusion_metric_distributions.png"
    
    with open(os.path.join(args.output_dir, summary_filename), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics summary saved to {os.path.join(args.output_dir, summary_filename)}")
    
    if per_sample_metrics_list:
        per_sample_df = pd.DataFrame(per_sample_metrics_list)
        csv_output_path = os.path.join(args.output_dir, csv_filename)
        per_sample_df.to_csv(csv_output_path, index=False)
        logger.info(f"Per-sample metrics saved to {csv_output_path}")

    # Plotting
    fig_height = 5
    num_plots = 0
    if all_ssim_scores: num_plots +=1
    if all_psnr_scores: num_plots +=1
    if all_feat_dists: num_plots +=1
    if all_batch_fids_list: num_plots +=1
    
    if num_plots > 0:
        plt.figure(figsize=(4 * num_plots, fig_height))
        plot_idx = 1

        if all_ssim_scores:
            plt.subplot(1, num_plots, plot_idx)
            plt.hist(all_ssim_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(ssim_mean, color='r', ls='dashed', lw=2, label=f'Mean: {ssim_mean:.3f}')
            plt.title('SSIM Distribution'); plt.xlabel('SSIM'); plt.ylabel('Count'); plt.legend()
            plot_idx+=1
        
        if all_psnr_scores:
            plt.subplot(1, num_plots, plot_idx)
            plt.hist(all_psnr_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.axvline(psnr_mean, color='r', ls='dashed', lw=2, label=f'Mean: {psnr_mean:.2f}')
            plt.title('PSNR Distribution'); plt.xlabel('PSNR (dB)'); plt.ylabel('Count'); plt.legend()
            plot_idx+=1

        if all_feat_dists:
            plt.subplot(1, num_plots, plot_idx)
            plt.hist(all_feat_dists, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.axvline(feat_dist_mean, color='r', ls='dashed', lw=2, label=f'Mean: {feat_dist_mean:.2f}')
            plt.title('Inception Feature Distance'); plt.xlabel('L2 Distance'); plt.ylabel('Count'); plt.legend()
            plot_idx+=1

        if all_batch_fids_list:
            plt.subplot(1, num_plots, plot_idx)
            # Ensure bins are reasonable for small numbers of batches
            num_bins = min(20, len(all_batch_fids_list) // 2 if len(all_batch_fids_list) > 4 else max(1, len(all_batch_fids_list)))
            if num_bins == 0 and len(all_batch_fids_list) > 0: num_bins = 1 # case for 1-4 batches
            if num_bins > 0 :
                plt.hist(all_batch_fids_list, bins=num_bins, alpha=0.7, color='gold', edgecolor='black')
                plt.axvline(batch_fid_mean, color='r', ls='dashed', lw=2, label=f'Mean: {batch_fid_mean:.2f}')
            else:
                 logger.warning("Not enough batch FID scores to plot distribution.")
            plt.title('Per-Batch FID Distribution'); plt.xlabel('FID'); plt.ylabel('Count'); plt.legend()
            plot_idx+=1
        
        plt.tight_layout()
        plot_output_path = os.path.join(args.output_dir, plot_filename)
        plt.savefig(plot_output_path)
        plt.close()
        logger.info(f"Metric distributions plot saved to {plot_output_path}")
    else:
        logger.info("No metrics available to plot.")

    logger.info(f"=== Evaluation Results (Aggregated) ===")
    if metrics_summary['num_samples_evaluated'] > 0 :
        logger.info(f"Number of samples evaluated: {metrics_summary['num_samples_evaluated']}")
        logger.info(f"Global FID Score (Dataset Level): {metrics_summary['global_fid'] if metrics_summary['global_fid'] is not None else 'N/A'}")
        logger.info(f"SSIM: {metrics_summary['ssim_mean']:.4f} +/- {metrics_summary['ssim_std']:.4f}" if metrics_summary['ssim_mean'] is not None else "SSIM: N/A")
        logger.info(f"PSNR: {metrics_summary['psnr_mean']:.4f} +/- {metrics_summary['psnr_std']:.4f}" if metrics_summary['psnr_mean'] is not None else "PSNR: N/A")
        logger.info(f"Per-Sample Inception Feature Distance: {metrics_summary['inception_feature_distance_mean']:.4f} +/- {metrics_summary['inception_feature_distance_std']:.4f}" if metrics_summary['inception_feature_distance_mean'] is not None else "Inception Feature Distance: N/A")
        if metrics_summary['num_valid_batch_fids'] > 0:
            logger.info(f"Per-Batch FID (Mean over {metrics_summary['num_valid_batch_fids']} batches): {metrics_summary['batch_fid_mean']:.4f} +/- {metrics_summary['batch_fid_std']:.4f}")
        else:
            logger.info("Per-Batch FID: N/A (no valid batches or all resulted in NaN)")
        logger.info(f"Stain Normalization: {'Enabled (' + args.stain_normalization_method + ')' if args.enable_stain_normalization else 'Disabled'}") # Added
        logger.info(f"Sampling Method: {args.sampling_method}")
    else:
        logger.info("No samples were evaluated.")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
