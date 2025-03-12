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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# For FID calculation
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from baseline.diffusion import GaussianDiffusion
from baseline.diffusion_train_ddp import generate_images_with_diffusion
from src.utils import setup_parser, parse_adata
from src.dataset import (CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn,
                        load_preprocessed_hest1k_singlecell_data,
                        OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn,
                        FastSeparatePatchDataset, fast_separate_patch_collate_fn)
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match

# Additional imports for UNI2-h
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import torchvision.transforms as transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import cv2
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, cohen_kappa_score
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def segment_nuclei(images):
    """Nuclear segmentation using basic image processing"""
    nuclei_masks = []
    
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to grayscale if needed
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = img.transpose(1, 2, 0)
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img[:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        # Otsu thresholding for nuclear segmentation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove small objects and fill holes
        binary = morphology.remove_small_objects(binary > 0, min_size=50)
        binary = morphology.remove_small_holes(binary, area_threshold=100)
        
        # Label connected components
        labeled = measure.label(binary)
        nuclei_masks.append(labeled)
    
    return nuclei_masks

def extract_nuclear_features(images, nuclei_masks):
    """Extract nuclear morphometric features"""
    features = {'area': [], 'perimeter': [], 'circularity': [], 'eccentricity': [], 'solidity': []}
    
    for img, mask in zip(images, nuclei_masks):
        regions = measure.regionprops(mask)
        
        for region in regions:
            # Basic morphometric features
            area = region.area
            perimeter = region.perimeter
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            eccentricity = region.eccentricity
            solidity = region.solidity
            
            features['area'].append(area)
            features['perimeter'].append(perimeter)
            features['circularity'].append(circularity)
            features['eccentricity'].append(eccentricity)
            features['solidity'].append(solidity)
    
    return features

def compare_distributions(real_values, gen_values, method='ks'):
    """Compare two distributions using statistical tests"""
    if len(real_values) == 0 or len(gen_values) == 0:
        return {'similarity': 0.0, 'p_value': 1.0}
    
    if method == 'ks':
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(real_values, gen_values)
        similarity = 1 - ks_stat  # Higher similarity for lower KS statistic
    elif method == 'wasserstein':
        # Wasserstein distance
        w_distance = wasserstein_distance(real_values, gen_values)
        similarity = 1 / (1 + w_distance)  # Convert distance to similarity
        p_value = None
    
    return {'similarity': similarity, 'p_value': p_value}

def calculate_classification_agreement(real_types, gen_types):
    """Calculate cell type classification agreement"""
    if len(real_types) != len(gen_types):
        return {'accuracy': 0.0, 'kappa': 0.0}
    
    # Handle single-label cases
    if len(np.unique(real_types)) == 1 or len(np.unique(gen_types)) == 1:
        return {'accuracy': 0.0, 'kappa': 0.0}
    
    try:
        accuracy = accuracy_score(real_types, gen_types)
        kappa = cohen_kappa_score(real_types, gen_types)
        
        # Handle NaN kappa values
        if np.isnan(kappa):
            kappa = 0.0
            
        return {'accuracy': accuracy, 'kappa': kappa}
    except Exception:
        return {'accuracy': 0.0, 'kappa': 0.0}

def load_uni2_h_model(device, model_path="/depot/natallah/data/Mengbo/HnE_RNA/GeneFlow/UNI2-h"):
    """Load UNI2-h foundation model using official timm approach"""    
    try:
        logger.info(f"Loading UNI2-h model using timm from {model_path}")
        
        # UNI2-h specific architecture parameters from official documentation
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        # Load model using local weights (since you have them downloaded)
        model = timm.create_model(
            'vit_giant_patch14_224',
            pretrained=False, 
            **timm_kwargs
        )
        
        # Load your local pytorch_model.bin
        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            logger.info("Successfully loaded UNI2-h weights from local pytorch_model.bin")
        else:
            raise FileNotFoundError(f"pytorch_model.bin not found at {pytorch_model_path}")
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        # Official UNI2-h transform
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        logger.info("UNI2-h model loaded successfully using official timm approach")
        return model, None, transform  # processor=None since we're using timm
        
    except Exception as e:
        logger.error(f"Failed to load UNI2-h model: {e}")
        raise RuntimeError(f"UNI2-h model loading failed: {e}. Cannot proceed without UNI2-h.")

def extract_uni2_h_embeddings(images, model, processor=None, preprocess_transform=None, device='cuda'):
    """Extract UNI2-h embeddings using official approach"""
    embeddings = []
    
    with torch.no_grad():
        for img in images:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            
            if img.dim() == 3:
                img = img.unsqueeze(0)
            
            # Ensure RGB format (UNI2-h expects 3 channels)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            elif img.shape[1] > 3:
                img = img[:, :3, :, :]
            
            img = img.to(device)
            
            # Apply official UNI2-h preprocessing
            if preprocess_transform is not None:
                # Convert tensor back to PIL for transform, then back to tensor
                img_pil = transforms.ToPILImage()(img.squeeze(0))
                img_transformed = preprocess_transform(img_pil).unsqueeze(0).to(device)
            else:
                # Fallback: manual normalization
                img_transformed = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                img_transformed = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img_transformed)
            
            # Extract features using timm model (returns [1, 1536] tensor)
            with torch.inference_mode():
                embedding = model(img_transformed)  # Shape: [1, 1536]
            
            embeddings.append(embedding.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

def classify_cell_types_uni2h(embeddings, threshold=0.7):
    """Enhanced cell type classification using UNI2-h embeddings"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Use PCA for dimensionality reduction if embeddings are high-dimensional
    if embeddings.shape[1] > 512:
        # Fix: Adaptive PCA components based on sample size
        max_components = min(256, embeddings.shape[0] - 1, embeddings.shape[1])
        if max_components > 1:
            pca = PCA(n_components=max_components, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
    else:
        embeddings_reduced = embeddings
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)
    
    # Adaptive clustering based on embedding quality
    n_samples = len(embeddings_scaled)
    if n_samples < 10:
        return np.arange(n_samples) % 2  # Alternate between 0 and 1 for variety
    
    # Use more clusters for UNI2-h as it can distinguish more cell types
    n_clusters = min(12, max(3, n_samples // 8))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cell_types = kmeans.fit_predict(embeddings_scaled)
    
    return cell_types

def detect_spatial_patterns_uni2h(images, model, processor=None, preprocess_transform=None, device='cuda'):
    """Enhanced spatial pattern detection using UNI2-h features"""
    patterns = []
    
    # Extract UNI2-h embeddings for spatial analysis
    embeddings = extract_uni2_h_embeddings(images, model, processor, preprocess_transform, device)
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.shape[0] in [3, 4]:
            img = img.transpose(1, 2, 0)
        
        # Convert to grayscale for traditional texture analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img[:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        # Compute texture features using GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        
        # Combine traditional texture features with UNI2-h embedding features
        embedding_features = embeddings[i] if i < len(embeddings) else np.zeros(512)
        
        pattern_features = {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'uni2h_spatial_complexity': np.std(embedding_features),  # Use embedding variance as spatial complexity measure
            'uni2h_feature_magnitude': np.mean(np.abs(embedding_features))  # Mean feature magnitude
        }
        
        patterns.append(pattern_features)
    
    return patterns

def extended_biological_evaluation_uni2h(real_images, generated_images, uni2h_model, processor=None, preprocess_transform=None, device='cuda'):
    """Comprehensive biological validation using UNI2-h"""
    results = {}
    
    # Convert tensors to numpy arrays if needed
    if isinstance(real_images, torch.Tensor):
        real_images = real_images.cpu().numpy()
    if isinstance(generated_images, torch.Tensor):
        generated_images = generated_images.cpu().numpy()
    
    try:
        # 1. Cell Type Classification Consistency using UNI2-h
        real_embeddings = extract_uni2_h_embeddings(real_images, uni2h_model, processor, preprocess_transform, device)
        gen_embeddings = extract_uni2_h_embeddings(generated_images, uni2h_model, processor, preprocess_transform, device)
        
        real_cell_types = classify_cell_types_uni2h(real_embeddings)
        gen_cell_types = classify_cell_types_uni2h(gen_embeddings)
        
        cell_type_agreement = calculate_classification_agreement(real_cell_types, gen_cell_types)
        results['cell_type_accuracy'] = cell_type_agreement['accuracy']
        results['cell_type_kappa'] = cell_type_agreement['kappa']
        
        # 2. UNI2-h Feature-based similarity
        embedding_similarity = compare_distributions(
            real_embeddings.flatten(), 
            gen_embeddings.flatten()
        )
        results['uni2h_embedding_similarity'] = embedding_similarity['similarity']
        results['uni2h_embedding_p_value'] = embedding_similarity.get('p_value', None)
        
        # 3. Nuclear Feature Analysis
        real_nuclei_masks = segment_nuclei(real_images)
        gen_nuclei_masks = segment_nuclei(generated_images)
        
        real_nuclear_features = extract_nuclear_features(real_images, real_nuclei_masks)
        gen_nuclear_features = extract_nuclear_features(generated_images, gen_nuclei_masks)
        
        # Compare nuclear features
        for feature_name in ['area', 'circularity', 'eccentricity', 'solidity']:
            if real_nuclear_features[feature_name] and gen_nuclear_features[feature_name]:
                comparison = compare_distributions(
                    real_nuclear_features[feature_name], 
                    gen_nuclear_features[feature_name]
                )
                results[f'nuclear_{feature_name}_similarity'] = comparison['similarity']
                results[f'nuclear_{feature_name}_p_value'] = comparison.get('p_value', None)
        
        # 4. Enhanced Tissue-Level Spatial Pattern Analysis with UNI2-h
        real_patterns = detect_spatial_patterns_uni2h(real_images, uni2h_model, processor, preprocess_transform, device)
        gen_patterns = detect_spatial_patterns_uni2h(generated_images, uni2h_model, processor, preprocess_transform, device)
        
        # Compare spatial patterns (including new UNI2-h features)
        for pattern_type in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']:
            real_pattern_values = [p[pattern_type] for p in real_patterns]
            gen_pattern_values = [p[pattern_type] for p in gen_patterns]
            
            pattern_comparison = compare_distributions(real_pattern_values, gen_pattern_values)
            results[f'spatial_{pattern_type}_similarity'] = pattern_comparison['similarity']
            results[f'spatial_{pattern_type}_p_value'] = pattern_comparison.get('p_value', None)
        
        # 5. Enhanced overall biological plausibility score
        bio_scores = [
            results.get('cell_type_accuracy', 0),
            results.get('uni2h_embedding_similarity', 0),
            np.mean([results.get(f'nuclear_{f}_similarity', 0) for f in ['area', 'circularity', 'eccentricity', 'solidity']]),
            np.mean([results.get(f'spatial_{f}_similarity', 0) for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']])
        ]
        results['overall_biological_plausibility'] = np.mean(bio_scores)
        
    except Exception as e:
        logger.warning(f"Error in UNI2-h biological evaluation: {e}")
        # Return default values if evaluation fails
        results = {
            'cell_type_accuracy': 0.0,
            'cell_type_kappa': 0.0,
            'uni2h_embedding_similarity': 0.0,
            'nuclear_area_similarity': 0.0,
            'nuclear_circularity_similarity': 0.0,
            'nuclear_eccentricity_similarity': 0.0,
            'nuclear_solidity_similarity': 0.0,
            'spatial_contrast_similarity': 0.0,
            'spatial_dissimilarity_similarity': 0.0,
            'spatial_homogeneity_similarity': 0.0,
            'spatial_energy_similarity': 0.0,
            'spatial_uni2h_spatial_complexity_similarity': 0.0,
            'spatial_uni2h_feature_magnitude_similarity': 0.0,
            'overall_biological_plausibility': 0.0
        }
    
    return results

def calculate_uni2h_fid(real_images, generated_images, uni2h_model, processor=None, preprocess_transform=None, device='cuda'):
    """Calculate FID using UNI2-h features instead of Inception"""
    try:
        real_features = extract_uni2_h_embeddings(real_images, uni2h_model, processor, preprocess_transform, device)
        gen_features = extract_uni2_h_embeddings(generated_images, uni2h_model, processor, preprocess_transform, device)
        
        uni2h_fid = calculate_fid(real_features, gen_features)
        return uni2h_fid
    except Exception as e:
        logger.warning(f"Error calculating UNI2-h FID: {e}")
        return np.nan

# ============================================================
# DDP Setup and Cleanup Functions
# ============================================================

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

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes for averaging"""
    if world_size == 1:
        return tensor
    
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor = reduced_tensor / world_size
    return reduced_tensor

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

# ============================================================
# Evaluation Models and Functions
# ============================================================

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

# ============================================================
# Main Evaluation Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA to H&E model with Diffusion using FID, SSIM, and PSNR metrics (DDP-enabled).")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained diffusion model.')
    parser.add_argument('--gene_expr', type=str, default="cell_256_aux/normalized.csv", help='Path to gene expression CSV file.')
    parser.add_argument('--image_paths', type=str, default="cell_256_aux/input/cell_image_paths.json", help='Path to JSON file with image paths.')
    parser.add_argument('--patch_image_paths', type=str, default="cell_256_aux/input/patch_image_paths.json", help='Path to JSON file with patch paths.')
    parser.add_argument('--patch_cell_mapping', type=str, default="cell_256_aux/input/patch_cell_mapping.json", help='Path to JSON file with mapping paths.')
    parser.add_argument('--output_dir', type=str, default='diffusion_eval_results', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation.')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the generated images.')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels.')
    parser.add_argument('--gen_steps', type=int, default=300, help='Number of steps for solver during generation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='single', help='Type of model to use: single-cell or multi-cell')
    parser.add_argument('--normalize_aux', action='store_true', help='Normalize auxiliary channels during dataset loading.')
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel evaluation.')
    parser.add_argument('--max_samples', type=int, default=int(1e4), help='Maximum number of samples to use from the dataset.')
    
    # Diffusion-specific arguments
    parser.add_argument('--diffusion_timesteps', type=int, default=300, help='Number of timesteps for diffusion process')
    parser.add_argument('--beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Noise schedule for diffusion')
    parser.add_argument('--predict_noise', action='store_true', default=True, help='Whether model predicts noise (True) or x_0 (False)')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method for diffusion generation')

    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--disable_ddp_debug', action='store_true', help='Disable DDP for debugging purposes')
    
    # HEST-1k arguments
    parser.add_argument('--hest1k_sid', type=str, nargs='*', default=None, help='HEST-1k sample ID for direct loading')
    parser.add_argument('--hest1k_base_dir', type=str, default=None, help='Base directory for HEST-1k data')
    parser.add_argument('--hest1k_xenium_dir', type=str, default=None, help='Directory for HEST-1k Xenium AnnData files')
    parser.add_argument('--hest1k_xenium_metadata', type=str, default=None, help='Metadata CSV for HEST-1k Xenium data')
    parser.add_argument('--hest1k_xenium_samples', type=str, nargs='*', default=None, help='Specific Xenium sample IDs to use')
    parser.add_argument('--hest1k_xenium_fast_dir', type=str, default=None,
                        help='Directory for reformatted fast-loading HEST-1k Xenium patch data')
    parser.add_argument('--num_dataloader_workers', type=int, default=4, help='Number of workers for data loading.')

    parser.add_argument('--save_embeddings', action='store_true', default=True, help='Save UNI2-h embeddings for later analysis')
    parser.add_argument('--embeddings_output_path', type=str, default=None, help='Path to save embeddings (if None, saves to output_dir)')

    # Arguments for stain normalization and others will be added by setup_parser
    parser = setup_parser(parser)
    args = parser.parse_args()

    if args.disable_ddp_debug:
        args.use_ddp = False

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
        logger.info(f"Evaluating DIFFUSION model with {args.sampling_method.upper()} sampling")
        logger.info(f"Generation steps: {args.gen_steps}, Diffusion timesteps: {args.diffusion_timesteps}")
        if hasattr(args, 'debug_mode') and args.debug_mode:
            logger.info("DEBUG MODE ENABLED")
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
        if uni2h_processor is not None:
            logger.info("UNI2-h model loaded successfully for biological validation")
        else:
            logger.info("UNI2-h model unavailable, using ResNet fallback")

    # Data loading logic (same as rectified flow)
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

    # Dataset creation logic (same as rectified flow)
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

    # Split into train and validation sets (same split as training)
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

    # Model Loading
    if rank == 0:
        logger.info(f"Loading pretrained diffusion model from {args.model_path}")
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        if rank == 0:
            logger.error(f"Model checkpoint not found at {args.model_path}")
        if args.use_ddp:
            cleanup_ddp()
        return

    model_state = checkpoint.get("model", checkpoint)
    model_config_ckpt = checkpoint.get("config", {})

    # Update img_channels from checkpoint if available
    args.img_channels = model_config_ckpt.get('img_channels', args.img_channels)

    model_constructor_args = dict(
        rna_dim=gene_dim,
        img_channels=args.img_channels,
        img_size=args.img_size,
        model_channels=model_config_ckpt.get('model_channels', 128),
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
        relation_rank=model_config_ckpt.get('relation_rank', getattr(args, 'relation_rank', 50))
    )
    
    if args.model_type == 'single':
        if rank == 0:
            logger.info("Initializing diffusion single-cell model")
        model = RNAtoHnEModel(**model_constructor_args)
    else:
        if rank == 0:
            logger.info("Initializing diffusion multi-cell model")
        model_constructor_args['num_aggregation_heads'] = model_config_ckpt.get('num_aggregation_heads', getattr(args, 'num_aggregation_heads', 4))
        model = MultiCellRNAtoHnEModel(**model_constructor_args)
    
    model.load_state_dict(model_state)
    model.to(device)

    # Wrap model with DDP if using distributed evaluation
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")

    model.eval()

    # Initialize diffusion and evaluation components
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

    all_real_embeddings_with_metadata = []
    all_gen_embeddings_with_metadata = []

    # UNI2-h FID calculation storage and biological validation
    all_uni2h_fids_list = []
    all_biological_results = []
    all_real_uni2h_features_for_fid = []
    all_gen_uni2h_features_for_fid = []

    if rank == 0:
        logger.info(f"Starting diffusion evaluation on {len(eval_loader)} batches")

    with torch.no_grad():
        # Create progress bar only on rank 0
        if rank == 0:
            eval_pbar = tqdm(eval_loader, desc="Evaluating Diffusion")
        else:
            eval_pbar = eval_loader

        for batch_idx, batch in enumerate(eval_pbar):
            sample_ids_in_batch = []
            real_images_tensor = batch['image'].to(device)
            current_batch_size = real_images_tensor.shape[0]

            try:
                if args.model_type == 'single':
                    gene_expr = batch['gene_expr'].to(device)
                    sample_ids_in_batch = batch['cell_id']
                    gene_mask = batch.get('gene_mask', None)
                    if gene_mask is not None:
                        gene_mask = gene_mask.to(device)

                    if hasattr(args, 'debug_mode') and args.debug_mode and rank == 0:
                        logger.info(f"Starting diffusion generation for batch {batch_idx} with {args.gen_steps} steps using {args.sampling_method}")

                    generated_images_tensor = generate_images_with_diffusion(
                        model=model, diffusion=diffusion, gene_expr=gene_expr, device=device,
                        num_steps=args.gen_steps, gene_mask=gene_mask, is_multi_cell=False, method=args.sampling_method
                    )
                    
                    if hasattr(args, 'debug_mode') and args.debug_mode and rank == 0:
                        logger.info(f"Completed diffusion generation for batch {batch_idx}")
                        
                else:  # multi-cell
                    processed_batch = prepare_multicell_batch(batch, device)
                    gene_expr = processed_batch['gene_expr']
                    num_cells_info = processed_batch.get('num_cells')
                    sample_ids_in_batch = batch['patch_id']
                    gene_mask = batch.get('gene_mask', None)
                    if gene_mask is not None:
                        gene_mask = gene_mask.to(device)

                    if hasattr(args, 'debug_mode') and args.debug_mode and rank == 0:
                        logger.info(f"Starting multi-cell diffusion generation for batch {batch_idx}")

                    generated_images_tensor = generate_images_with_diffusion(
                        model=model, diffusion=diffusion, gene_expr=gene_expr, device=device,
                        num_steps=args.gen_steps, num_cells=num_cells_info, gene_mask=gene_mask, is_multi_cell=True, method=args.sampling_method
                    )
                    
                    if hasattr(args, 'debug_mode') and args.debug_mode and rank == 0:
                        logger.info(f"Completed multi-cell diffusion generation for batch {batch_idx}")

            except Exception as e:
                logger.error(f"Error during diffusion generation in batch {batch_idx}: {e}")
                if hasattr(args, 'debug_mode') and args.debug_mode:
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise
                # Skip this batch and continue
                continue

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
            if real_features.shape[0] > 0:
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
                eval_pbar.set_postfix({
                    'Avg SSIM': f'{np.mean(all_ssim_scores):.4f}',
                    'Avg PSNR': f'{np.mean(all_psnr_scores):.4f}',
                    'Inception FID': f'{batch_fid:.2f}' if not np.isnan(batch_fid) else 'N/A',
                    'UNI2-h FID': f'{uni2h_fid:.2f}' if not np.isnan(uni2h_fid) else 'N/A'
                })

            # Early exit for debugging
            if hasattr(args, 'debug_mode') and args.debug_mode and batch_idx >= 2:
                if rank == 0:
                    logger.info("DEBUG MODE: Stopping after 3 batches")
                break

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
        logger.info("Computing final diffusion evaluation metrics...")
        
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

        # Enhanced results summary with biological validation
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
            'model_type': args.model_type,
            'img_size': args.img_size,
            'img_channels': args.img_channels,
            'generation_steps': args.gen_steps,
            'batch_size': original_batch_size,
            'world_size': world_size,
            'inception_feature_distance_mean': float(feat_dist_mean) if not np.isnan(feat_dist_mean) else None,
            'inception_feature_distance_std': float(feat_dist_std) if not np.isnan(feat_dist_std) else None,
            'method': 'diffusion',
            'sampling_method': args.sampling_method,
            'diffusion_timesteps': args.diffusion_timesteps,
            'beta_schedule': args.beta_schedule
        }
        
        # Add biological metrics to results summary
        results_summary.update(biological_summary)

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
            np.save(os.path.join(embeddings_dir, 'diffusion_uni2h_embeddings.npy'), embeddings)
            
            # Save metadata
            metadata_df = pd.DataFrame({
                'sample_id': sample_ids,
                'type': types,
                'batch_idx': batch_indices,
                'embedding_index': range(len(sample_ids)),
                'method': 'diffusion'
            })
            metadata_df.to_csv(os.path.join(embeddings_dir, 'diffusion_embeddings_metadata.csv'), index=False)
            
            # Also save separate arrays for convenience
            real_embeddings = np.array([entry['embedding'] for entry in all_real_embeddings_with_metadata])
            gen_embeddings = np.array([entry['embedding'] for entry in all_gen_embeddings_with_metadata])
            
            np.save(os.path.join(embeddings_dir, 'diffusion_uni2h_real_embeddings.npy'), real_embeddings)
            np.save(os.path.join(embeddings_dir, 'diffusion_uni2h_generated_embeddings.npy'), gen_embeddings)
            
            # Save sample IDs separately
            real_sample_ids = [entry['sample_id'] for entry in all_real_embeddings_with_metadata]
            gen_sample_ids = [entry['sample_id'] for entry in all_gen_embeddings_with_metadata]
            
            with open(os.path.join(embeddings_dir, 'diffusion_real_sample_ids.json'), 'w') as f:
                json.dump(real_sample_ids, f)
            with open(os.path.join(embeddings_dir, 'diffusion_generated_sample_ids.json'), 'w') as f:
                json.dump(gen_sample_ids, f)
            
            logger.info(f"Saved {embeddings.shape[0]} diffusion UNI2-h embeddings ({embeddings.shape[1]} dimensions) to {embeddings_dir}")
            logger.info(f"Real embeddings: {real_embeddings.shape[0]}, Generated embeddings: {gen_embeddings.shape[0]}")

        # Enhanced logging with UNI2-h and biological validation results
        logger.info("="*80)
        logger.info("DIFFUSION MODEL EVALUATION RESULTS WITH UNI2-H BIOLOGICAL VALIDATION")
        logger.info(f"(DDPM/DDIM Sampling - {args.sampling_method.upper()} method)")
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
            logger.info(f"Enhanced Spatial Pattern Similarity (avg): {np.mean([biological_summary.get(f'mean_spatial_{f}_similarity', 0) for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']]):.4f}")
            logger.info(f"Overall Biological Plausibility: {biological_summary.get('mean_overall_biological_plausibility', 0):.4f}")
        logger.info("="*80)

        # Save summary to JSON
        with open(os.path.join(args.output_dir, 'diffusion_evaluation_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)

        # Save per-sample metrics to CSV
        per_sample_df = pd.DataFrame(per_sample_metrics_list)
        per_sample_df.to_csv(os.path.join(args.output_dir, 'diffusion_per_sample_metrics.csv'), index=False)

        # Save batch-level biological validation metrics separately
        biological_df = pd.DataFrame(all_biological_results)
        biological_df.to_csv(os.path.join(args.output_dir, 'diffusion_batch_level_biological_validation.csv'), index=False)

        # Create and save enhanced plots
        plt.figure(figsize=(20, 10))
        
        # Row 1: Traditional metrics
        plt.subplot(2, 4, 1)
        plt.hist(all_ssim_scores, bins=50, alpha=0.7, color='blue')
        plt.axvline(mean_ssim, color='red', linestyle='--', label=f'Mean: {mean_ssim:.4f}')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.title('SSIM Distribution (Diffusion)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 4, 2)
        plt.hist(all_psnr_scores, bins=50, alpha=0.7, color='green')
        plt.axvline(mean_psnr, color='red', linestyle='--', label=f'Mean: {mean_psnr:.4f}')
        plt.xlabel('PSNR')
        plt.ylabel('Frequency')
        plt.title('PSNR Distribution (Diffusion)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 4, 3)
        if valid_fids:
            plt.hist(valid_fids, bins=50, alpha=0.7, color='orange')
            plt.axvline(mean_batch_fid, color='red', linestyle='--', label=f'Mean: {mean_batch_fid:.4f}')
        plt.xlabel('Inception FID')
        plt.ylabel('Frequency')
        plt.title('Inception FID Distribution (Diffusion)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 4, 4)
        if valid_uni2h_fids:
            plt.hist(valid_uni2h_fids, bins=50, alpha=0.7, color='purple')
            plt.axvline(mean_uni2h_fid, color='red', linestyle='--', label=f'Mean: {mean_uni2h_fid:.4f}')
        plt.xlabel('UNI2-H FID')
        plt.ylabel('Frequency')
        plt.title('UNI2-H FID Distribution (Diffusion)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Row 2: Biological validation metrics
        if all_biological_results:
            cell_type_accuracies = [r['cell_type_accuracy'] for r in all_biological_results if 'cell_type_accuracy' in r and not np.isnan(r['cell_type_accuracy'])]
            plt.subplot(2, 4, 5)
            if cell_type_accuracies:
                plt.hist(cell_type_accuracies, bins=30, alpha=0.7, color='cyan')
                plt.axvline(np.mean(cell_type_accuracies), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(cell_type_accuracies):.4f}')
            plt.xlabel('Cell Type Accuracy')
            plt.ylabel('Frequency')
            plt.title('Cell Type Classification (Diffusion)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            uni2h_similarities = [r['uni2h_embedding_similarity'] for r in all_biological_results if 'uni2h_embedding_similarity' in r and not np.isnan(r['uni2h_embedding_similarity'])]
            plt.subplot(2, 4, 6)
            if uni2h_similarities:
                plt.hist(uni2h_similarities, bins=30, alpha=0.7, color='magenta')
                plt.axvline(np.mean(uni2h_similarities), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(uni2h_similarities):.4f}')
            plt.xlabel('UNI2-H Embedding Similarity')
            plt.ylabel('Frequency')
            plt.title('UNI2-H Feature Similarity (Diffusion)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            nuclear_areas = [r['nuclear_area_similarity'] for r in all_biological_results if 'nuclear_area_similarity' in r and not np.isnan(r['nuclear_area_similarity'])]
            plt.subplot(2, 4, 7)
            if nuclear_areas:
                plt.hist(nuclear_areas, bins=30, alpha=0.7, color='brown')
                plt.axvline(np.mean(nuclear_areas), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(nuclear_areas):.4f}')
            plt.xlabel('Nuclear Area Similarity')
            plt.ylabel('Frequency')
            plt.title('Nuclear Morphometry (Diffusion)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            bio_plausibilities = [r['overall_biological_plausibility'] for r in all_biological_results if 'overall_biological_plausibility' in r and not np.isnan(r['overall_biological_plausibility'])]
            plt.subplot(2, 4, 8)
            if bio_plausibilities:
                plt.hist(bio_plausibilities, bins=30, alpha=0.7, color='gold')
                plt.axvline(np.mean(bio_plausibilities), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(bio_plausibilities):.4f}')
            plt.xlabel('Overall Biological Plausibility')
            plt.ylabel('Frequency')
            plt.title('Biological Validation Score (Diffusion)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'diffusion_evaluation_metrics_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        
        logger.info(f"Diffusion evaluation results saved to {args.output_dir}")

    # Cleanup DDP
    if args.use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()