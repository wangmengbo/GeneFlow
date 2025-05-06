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
from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata
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
        # Remove the fully connected layer
        self.inception_model.fc = torch.nn.Identity()
        # No need for gradients
        for param in self.inception_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Resize the input to the required shape
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Ensure we're using RGB channels (first 3 channels if more are available)
        if x.shape[1] > 3:
            x = x[:, :3]
        
        # Normalize to match InceptionV3 expected input range
        x = (x * 2) - 1
        
        # Get features
        x = self.inception_model(x)
        
        return x

# Calculate FID score
def calculate_fid(real_features, gen_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical error might lead to complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Calculate SSIM and PSNR for a batch of images
def calculate_image_metrics(real_images, generated_images):
    batch_size = real_images.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        # Extract real and generated images
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Calculate SSIM for RGB channels
        ssim_score = ssim(
            real_img[:,:,:3], 
            gen_img[:,:,:3], 
            channel_axis=2, 
            data_range=1.0
        )
        
        # Calculate PSNR for RGB channels
        psnr_score = psnr(
            real_img[:,:,:3], 
            gen_img[:,:,:3], 
            data_range=1.0
        )
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return ssim_scores, psnr_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA to H&E model with Rectified Flow using FID, SSIM, and PSNR metrics.")
    parser.add_argument('--model_path', type=str, default="cell_256_aux/output_rectified/best_single_rna_to_hne_model_rectified-multihead1.pt", help='Path to the pretrained model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='cell_256_aux/output_rectified', help='Directory to save outputs.')
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load gene expression data
    if args.adata is not None:
        logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols = parse_adata(args)
    else:
        logger.info(f"Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
    logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")

    # Create appropriate dataset based on model type
    if args.model_type == 'single':
        logger.info("Creating single-cell dataset")

        # Load image paths if provided (for visualization)
        if args.image_paths:
            logger.info(f"Loading image paths from {args.image_paths}")
            with open(args.image_paths, "r") as f:
                image_paths = json.load(f)
            logger.info(f"Loaded {len(image_paths)} cell image paths")

            # Filter out non-existent files
            image_paths_tmp = {}
            for k,v in image_paths.items():
                if os.path.exists(v):
                    image_paths_tmp[k] = v
            
            image_paths = image_paths_tmp
            logger.info(f"After filtering: {len(image_paths)} valid cell image paths")
        else:
            image_paths = {}
        
        dataset = CellImageGeneDataset(
            expr_df, 
            image_paths, 
            img_size=args.img_size,
            img_channels=args.img_channels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]),
            missing_gene_symbols=missing_gene_symbols if 'missing_gene_symbols' in locals() else None,
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
                patch_image_paths = json.load(f)

            # Filter out non-existent files
            patch_image_paths_tmp = {}
            for k,v in patch_image_paths.items():
                if os.path.exists(v):
                    patch_image_paths_tmp[k] = v
            patch_image_paths = patch_image_paths_tmp
            logger.info(f"After filtering: {len(patch_image_paths)} valid patch image paths")
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
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Create validation set
    # If specific number of evaluation samples is provided, use that
    if args.eval_samples is not None:
        eval_size = min(args.eval_samples, len(dataset))
        eval_indices = torch.randperm(len(dataset))[:eval_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
    else:
        # Otherwise use 20% of the data for evaluation (or the whole dataset if small)
        eval_size = min(int(0.2 * len(dataset)), len(dataset))
        eval_indices = torch.randperm(len(dataset))[:eval_size]
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
    
    # Create data loader
    if args.model_type == 'multi':
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=patch_collate_fn
        )
    else:
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    logger.info(f"Evaluation set size: {len(eval_dataset)}")
    
    # Initialize appropriate model based on model type
    gene_dim = expr_df.shape[1]  # Number of genes
    
    if args.model_type == 'single':
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
    
    # Load the pretrained model
    logger.info(f"Loading pretrained model from {args.model_path}")
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Initialize the rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    logger.info("Initialized rectified flow")
    
    # Initialize the inception model for FID calculation
    inception_model = InceptionModel(device)
    
    # Lists to store metrics
    all_ssim = []
    all_psnr = []
    all_real_features = []
    all_gen_features = []
    
    # Process each batch
    logger.info(f"Starting evaluation on {len(eval_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Handle data differently based on model type
            if args.model_type == 'single':
                gene_expr = batch['gene_expr'].to(device)
                real_images = batch['image'].to(device)
                cell_ids = batch['cell_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)
                
                # Generate images with rectified flow
                generated_images = generate_images_with_rectified_flow(
                    model=model,
                    rectified_flow=rectified_flow,
                    gene_expr=gene_expr,
                    device=device,
                    num_steps=args.gen_steps,
                    gene_mask=gene_mask,
                    is_multi_cell=False
                )
            else:  # multi-cell model
                # Prepare batch for multi-cell model
                processed_batch = prepare_multicell_batch(batch, device)
                gene_expr = processed_batch['gene_expr']
                num_cells = processed_batch['num_cells']
                real_images = batch['image'].to(device)
                
                # Generate images with rectified flow
                generated_images = generate_images_with_rectified_flow(
                    model=model,
                    rectified_flow=rectified_flow,
                    gene_expr=gene_expr,
                    device=device,
                    num_steps=args.gen_steps,
                    num_cells=num_cells,
                    is_multi_cell=True
                )
            
            # Extract features for FID calculation
            real_features = inception_model(real_images)
            gen_features = inception_model(generated_images)
            
            all_real_features.append(real_features.cpu().numpy())
            all_gen_features.append(gen_features.cpu().numpy())
            
            # Calculate SSIM and PSNR
            ssim_scores, psnr_scores = calculate_image_metrics(real_images, generated_images)
            all_ssim.extend(ssim_scores)
            all_psnr.extend(psnr_scores)
    
    # Concatenate all features
    all_real_features = np.concatenate(all_real_features, axis=0)
    all_gen_features = np.concatenate(all_gen_features, axis=0)
    
    # Calculate FID
    fid_score = calculate_fid(all_real_features, all_gen_features)
    
    # Calculate mean and std of SSIM and PSNR
    ssim_mean, ssim_std = np.mean(all_ssim), np.std(all_ssim)
    psnr_mean, psnr_std = np.mean(all_psnr), np.std(all_psnr)
    
    # Save metrics to file
    metrics = {
        'fid': float(fid_score),
        'ssim_mean': float(ssim_mean),
        'ssim_std': float(ssim_std),
        'psnr_mean': float(psnr_mean),
        'psnr_std': float(psnr_std),
        'num_samples': len(all_ssim)
    }
    
    with open(os.path.join(args.output_dir, 'rectified_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save distributions of SSIM and PSNR
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_ssim, bins=20, alpha=0.7)
    plt.axvline(ssim_mean, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'SSIM Distribution (Mean: {ssim_mean:.4f})')
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_psnr, bins=20, alpha=0.7)
    plt.axvline(psnr_mean, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'PSNR Distribution (Mean: {psnr_mean:.4f})')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'rectified_metric_distributions.png'))
    
    # Print summary
    logger.info(f"=== Evaluation Results ===")
    logger.info(f"Number of samples: {len(all_ssim)}")
    logger.info(f"FID Score: {fid_score:.4f}")
    logger.info(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    logger.info(f"PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()