import os
import json
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import cv2
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, cohen_kappa_score
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


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
    
    accuracy = accuracy_score(real_types, gen_types)
    kappa = cohen_kappa_score(real_types, gen_types)
    
    return {'accuracy': accuracy, 'kappa': kappa}


def classify_cell_types_uni2h(embeddings, threshold=0.7):
    """cell type classification using UNI2-h embeddings"""
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
        return np.zeros(n_samples)
    
    # Use more clusters for UNI2-h as it can distinguish more cell types
    n_clusters = min(12, max(3, n_samples // 8))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cell_types = kmeans.fit_predict(embeddings_scaled)
    
    return cell_types


def detect_spatial_patterns_uni2h(images, model, processor=None, preprocess_transform=None, device='cuda'):
    """spatial pattern detection using UNI2-h features"""
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
        
        # 4. Tissue-Level Spatial Pattern Analysis with UNI2-h
        real_patterns = detect_spatial_patterns_uni2h(real_images, uni2h_model, processor, preprocess_transform, device)
        gen_patterns = detect_spatial_patterns_uni2h(generated_images, uni2h_model, processor, preprocess_transform, device)
        
        # Compare spatial patterns (including new UNI2-h features)
        for pattern_type in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']:
            real_pattern_values = [p[pattern_type] for p in real_patterns]
            gen_pattern_values = [p[pattern_type] for p in gen_patterns]
            
            pattern_comparison = compare_distributions(real_pattern_values, gen_pattern_values)
            results[f'spatial_{pattern_type}_similarity'] = pattern_comparison['similarity']
            results[f'spatial_{pattern_type}_p_value'] = pattern_comparison.get('p_value', None)
        
        # 5. overall biological plausibility score
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
        
        uni2h_fid = calculate_fid_from_features(real_features, gen_features)
        return uni2h_fid
    except Exception as e:
        logger.warning(f"Error calculating UNI2-h FID: {e}")
        return np.nan


def calculate_fid_from_features(real_features, gen_features):
    """Calculate FID from precomputed features"""
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