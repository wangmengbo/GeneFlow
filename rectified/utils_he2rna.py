import sys
import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet50
from einops import rearrange
import safetensors.torch
from sequoia.src.he2rna import HE2RNA

logger = logging.getLogger(__name__)

def load_he2rna_model(model_path, device):
    """Load pretrained HE2RNA model from safetensors format"""
    try:        
        # Check if it's a HuggingFace model directory with safetensors
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            config_path = os.path.join(model_path, "config.json")
            
            if os.path.exists(safetensors_path):
                logger.info(f"Found safetensors model at: {safetensors_path}")
                
                # Try HuggingFace format first if config exists
                if os.path.exists(config_path):
                    try:
                        he2rna_model = HE2RNA.from_pretrained(model_path)
                        he2rna_model.to(device)
                        he2rna_model.eval()
                        logger.info(f"HE2RNA model loaded from HuggingFace format: {model_path}")
                        return he2rna_model
                    except Exception as e:
                        logger.warning(f"HuggingFace loading failed: {e}. Trying manual loading...")
                
                # Manual loading when no config.json
                logger.info("No config.json found or HuggingFace loading failed. Loading manually...")
                
                # Load state dict from safetensors
                state_dict = safetensors.torch.load_file(safetensors_path)
                
                # Infer model parameters from state dict
                # Check the output dimension from the last layer
                output_dim = None
                for key in state_dict.keys():
                    if 'conv2.bias' in key or 'layers.-1.bias' in key:
                        output_dim = state_dict[key].shape[0]
                        break
                
                if output_dim is None:
                    # Look for any conv layer with bias to get output dim
                    for key in state_dict.keys():
                        if 'conv' in key and 'bias' in key:
                            output_dim = state_dict[key].shape[0]
                            logger.info(f"Inferred output_dim from {key}: {output_dim}")
                            break
                
                if output_dim is None:
                    logger.warning("Could not infer output dimension from state dict")
                    output_dim = 19198  # Default from BRCA dataset
                
                # Create model with default parameters (matching he2rna.py)
                he2rna_model = HE2RNA(
                    input_dim=2048, 
                    layers=[256, 256],
                    ks=[1, 2, 5, 10, 20, 50, 100],
                    output_dim=output_dim, 
                    device=device
                )
                
                # Load state dict
                he2rna_model.load_state_dict(state_dict)
                he2rna_model.eval()
                logger.info(f"HE2RNA model loaded manually from safetensors with output_dim={output_dim}")
                return he2rna_model
                
            else:
                logger.warning(f"model.safetensors not found in directory: {model_path}")
                # List available files for debugging
                available_files = os.listdir(model_path)
                logger.info(f"Available files in {model_path}: {available_files}")
                return None
                
        # Check if it's a direct safetensors file path
        elif os.path.isfile(model_path) and model_path.endswith('.safetensors'):
            logger.info(f"Loading HE2RNA model from safetensors file: {model_path}")
            
            # Load state dict from safetensors
            state_dict = safetensors.torch.load_file(model_path)
            
            # Infer output dimension
            output_dim = None
            for key in state_dict.keys():
                if 'conv2.bias' in key or 'layers.-1.bias' in key:
                    output_dim = state_dict[key].shape[0]
                    break
                    
            if output_dim is None:
                output_dim = 19198  # Default from BRCA dataset
            
            # Create model with default parameters
            he2rna_model = HE2RNA(
                input_dim=2048, 
                layers=[256, 256],
                ks=[1, 2, 5, 10, 20, 50, 100],
                output_dim=output_dim, 
                device=device
            )
            
            # Load state dict
            he2rna_model.load_state_dict(state_dict)
            he2rna_model.eval()
            logger.info(f"HE2RNA model loaded from safetensors file with output_dim={output_dim}")
            return he2rna_model
            
        else:
            logger.warning(f"Model path does not exist or is not in expected format: {model_path}")
            return None
            
    except ImportError as e:
        logger.warning(f"Could not import required modules: {e}")
        logger.warning("Make sure safetensors and src.he2rna are available: pip install safetensors")
        return None
    except Exception as e:
        logger.warning(f"Could not load HE2RNA model from {model_path}: {e}")
        logger.warning(f"Error type: {type(e).__name__}")
        logger.warning(f"Error details: {str(e)}")
        return None

def extract_resnet_features_for_he2rna(images, device):
    """Extract ResNet features for HE2RNA (same as HE2RNA training)"""
    # Load the same ResNet model used for HE2RNA training
    feature_extractor = resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features = []
    with torch.no_grad():
        for img in images:
            # Convert to RGB if needed (HE2RNA expects RGB)
            if img.shape[0] > 3:
                rgb_img = img[:3]  # Take first 3 channels
            elif img.shape[0] == 1:
                rgb_img = img.repeat(3, 1, 1)
            else:
                rgb_img = img
            
            # Resize to expected input size
            rgb_img = F.interpolate(rgb_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            
            # Extract features
            feat = feature_extractor(rgb_img)  # Shape: [1, 2048, 1, 1]
            feat = feat.view(1, 2048, 1)  # Reshape for HE2RNA: [1, 2048, 1]
            features.append(feat)
    
    return torch.cat(features, dim=0)  # Shape: [batch, 2048, 1]

def predict_and_compare_rna_from_images(real_images, generated_images, he2rna_model, device):
    """
    Predict RNA from both real and generated images, then compare them.
    This is more fair than comparing against original RNA expression data.
    """
    if he2rna_model is None:
        return None, None, None
    
    try:
        # Predict RNA from real images
        real_rna_predictions, real_features = predict_rna_from_images(real_images, he2rna_model, device)
        
        # Predict RNA from generated images  
        gen_rna_predictions, gen_features = predict_rna_from_images(generated_images, he2rna_model, device)
        
        if real_rna_predictions is None or gen_rna_predictions is None:
            return None, None, None
            
        return real_rna_predictions, gen_rna_predictions, (real_features, gen_features)
        
    except Exception as e:
        logger.warning(f"Error in RNA prediction comparison: {e}")
        return None, None, None

def predict_rna_from_images(images, he2rna_model, device):
    """Helper function to predict RNA from any set of images"""
    if he2rna_model is None:
        return None, None
    
    try:
        # Extract features compatible with HE2RNA
        features = extract_resnet_features_for_he2rna(images, device)
        
        # Rearrange dimensions for HE2RNA
        features = features.permute(0, 2, 1)  # [batch, 1, 2048]
        features = rearrange(features, 'b t f -> b f t')
        
        # Temporarily modify the model's ks parameter to only use k=1 for single images
        original_ks = he2rna_model.ks.copy()
        he2rna_model.ks = np.array([1])
        
        # Get RNA predictions
        with torch.no_grad():
            rna_predictions = he2rna_model(features)
            rna_predictions = torch.nn.ReLU()(rna_predictions)
        
        # Restore original ks
        he2rna_model.ks = original_ks
        
        return rna_predictions, features
        
    except Exception as e:
        logger.warning(f"Error in RNA prediction: {e}")
        return None, None

def debug_rna_prediction_comparison(real_images, generated_images, he2rna_model, device):
    """Debug function to check what's happening with RNA predictions"""
    import torch
    import numpy as np
    from scipy.stats import pearsonr
    
    logger.info("=== DEBUGGING RNA PREDICTION COMPARISON ===")
    
    # Check if images are different
    if torch.equal(real_images, generated_images):
        logger.warning("WARNING: Real and generated images are IDENTICAL!")
        return None, None, None
    else:
        img_diff = torch.mean(torch.abs(real_images - generated_images)).item()
        logger.info(f"Image difference (MAE): {img_diff:.6f}")
    
    # Extract features and check if they're different
    real_features = extract_resnet_features_for_he2rna(real_images, device)
    gen_features = extract_resnet_features_for_he2rna(generated_images, device)
    
    if torch.equal(real_features, gen_features):
        logger.warning("WARNING: Extracted features are IDENTICAL!")
        return None, None, None
    else:
        feat_diff = torch.mean(torch.abs(real_features - gen_features)).item()
        logger.info(f"Feature difference (MAE): {feat_diff:.6f}")
    
    # Predict RNA and check differences
    real_rna_pred, _ = predict_rna_from_images(real_images, he2rna_model, device)
    gen_rna_pred, _ = predict_rna_from_images(generated_images, he2rna_model, device)
    
    if real_rna_pred is None or gen_rna_pred is None:
        logger.warning("RNA prediction failed")
        return None, None, None
    
    if torch.equal(real_rna_pred, gen_rna_pred):
        logger.warning("WARNING: RNA predictions are IDENTICAL!")
        return None, None, None
    else:
        rna_diff = torch.mean(torch.abs(real_rna_pred - gen_rna_pred)).item()
        logger.info(f"RNA prediction difference (MAE): {rna_diff:.6f}")
    
    # Convert to numpy for correlation analysis
    real_rna_np = real_rna_pred.cpu().numpy()
    gen_rna_np = gen_rna_pred.cpu().numpy()
    
    logger.info(f"Real RNA shape: {real_rna_np.shape}")
    logger.info(f"Generated RNA shape: {gen_rna_np.shape}")
    logger.info(f"Real RNA stats: mean={np.mean(real_rna_np):.4f}, std={np.std(real_rna_np):.4f}")
    logger.info(f"Gen RNA stats: mean={np.mean(gen_rna_np):.4f}, std={np.std(gen_rna_np):.4f}")
    
    # Calculate sample-wise correlations with detailed logging
    sample_correlations = []
    for i in range(min(3, real_rna_np.shape[0])):  # Check first 3 samples
        real_sample = real_rna_np[i]
        gen_sample = gen_rna_np[i]
        
        logger.info(f"Sample {i}:")
        logger.info(f"  Real RNA stats: mean={np.mean(real_sample):.4f}, std={np.std(real_sample):.4f}")
        logger.info(f"  Gen RNA stats: mean={np.mean(gen_sample):.4f}, std={np.std(gen_sample):.4f}")
        logger.info(f"  Sample difference (MAE): {np.mean(np.abs(real_sample - gen_sample)):.6f}")
        
        # Check if either sample has zero variance
        if np.std(real_sample) == 0 or np.std(gen_sample) == 0:
            logger.warning(f"  Sample {i}: Zero variance detected!")
            logger.info(f"    Real sample unique values: {len(np.unique(real_sample))}")
            logger.info(f"    Gen sample unique values: {len(np.unique(gen_sample))}")
            continue
            
        # Calculate correlation using both numpy and scipy
        corr_numpy = np.corrcoef(real_sample, gen_sample)[0, 1]
        corr_scipy, _ = pearsonr(real_sample, gen_sample)
        
        logger.info(f"  Correlation (numpy): {corr_numpy:.6f}")
        logger.info(f"  Correlation (scipy): {corr_scipy:.6f}")
        
        sample_correlations.append(corr_numpy)
    
    logger.info(f"Sample correlations: {sample_correlations}")
    logger.info("=== END DEBUG ===")
    
    return real_rna_pred, gen_rna_pred, None

def calculate_rna_image_comparison_metrics(real_rna_predictions, gen_rna_predictions):
    """
    Calculate metrics comparing RNA predictions from real vs generated images.
    This is a more fair biological validation than comparing against original RNA.
    """
    if real_rna_predictions is None or gen_rna_predictions is None:
        return {}
    
    try:
        # Convert to numpy
        if isinstance(real_rna_predictions, torch.Tensor):
            real_rna_predictions = real_rna_predictions.cpu().numpy()
        if isinstance(gen_rna_predictions, torch.Tensor):
            gen_rna_predictions = gen_rna_predictions.cpu().numpy()
        
        # Ensure same shape
        min_samples = min(real_rna_predictions.shape[0], gen_rna_predictions.shape[0])
        real_rna_predictions = real_rna_predictions[:min_samples]
        gen_rna_predictions = gen_rna_predictions[:min_samples]
        
        # Add debugging for the first calculation
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"RNA comparison shapes: real={real_rna_predictions.shape}, gen={gen_rna_predictions.shape}")
            logger.debug(f"First sample real RNA: mean={np.mean(real_rna_predictions[0]):.4f}, std={np.std(real_rna_predictions[0]):.4f}")
            logger.debug(f"First sample gen RNA: mean={np.mean(gen_rna_predictions[0]):.4f}, std={np.std(gen_rna_predictions[0]):.4f}")
        
        # Calculate correlation per sample (biological consistency per cell/patch)
        sample_correlations = []
        sample_mse_values = []
        
        for i in range(real_rna_predictions.shape[0]):
            real_sample = real_rna_predictions[i]
            gen_sample = gen_rna_predictions[i]
            
            # Skip samples with zero variance (constant predictions)
            if np.std(real_sample) == 0 or np.std(gen_sample) == 0:
                logger.warning(f"Sample {i}: Skipping due to zero variance (constant predictions)")
                continue
            
            # Per-sample correlation between real and generated RNA predictions
            corr = np.corrcoef(real_sample, gen_sample)[0, 1]
            if not np.isnan(corr):
                sample_correlations.append(corr)
            
            # Per-sample MSE
            mse = np.mean((real_sample - gen_sample) ** 2)
            sample_mse_values.append(mse)
        
        # Calculate gene-wise correlations (how consistent each gene's prediction is)
        gene_correlations = []
        for gene_idx in range(real_rna_predictions.shape[1]):
            real_gene = real_rna_predictions[:, gene_idx]
            gen_gene = gen_rna_predictions[:, gene_idx]
            
            # Skip genes with zero variance
            if np.std(real_gene) == 0 or np.std(gen_gene) == 0:
                continue
                
            if len(np.unique(real_gene)) > 1:  # Only if gene has variance
                corr = np.corrcoef(real_gene, gen_gene)[0, 1]
                if not np.isnan(corr):
                    gene_correlations.append(corr)
        
        # Overall correlation across all samples and genes
        real_flat = real_rna_predictions.flatten()
        gen_flat = gen_rna_predictions.flatten()
        
        # Check for variance in flattened arrays
        if np.std(real_flat) == 0 or np.std(gen_flat) == 0:
            overall_corr = 0.0
            logger.warning("Overall correlation set to 0 due to zero variance in flattened predictions")
        else:
            overall_corr = np.corrcoef(real_flat, gen_flat)[0, 1] if len(np.unique(real_flat)) > 1 else 0.0
        
        return {
            'rna_image_sample_correlation_mean': np.mean(sample_correlations) if sample_correlations else 0.0,
            'rna_image_sample_correlation_std': np.std(sample_correlations) if sample_correlations else 0.0,
            'rna_image_gene_correlation_mean': np.mean(gene_correlations) if gene_correlations else 0.0,
            'rna_image_gene_correlation_std': np.std(gene_correlations) if gene_correlations else 0.0,
            'rna_image_overall_correlation': float(overall_corr) if not np.isnan(overall_corr) else 0.0,
            'rna_image_mse_mean': np.mean(sample_mse_values),
            'rna_image_mse_std': np.std(sample_mse_values),
            'num_valid_sample_correlations': len(sample_correlations),
            'num_valid_gene_correlations': len(gene_correlations),
            'genes_compared': real_rna_predictions.shape[1]
        }
        
    except Exception as e:
        logger.warning(f"Error calculating RNA image comparison metrics: {e}")
        return {}