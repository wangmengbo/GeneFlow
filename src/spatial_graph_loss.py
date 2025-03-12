import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: cellpose not available. Install with: pip install cellpose")
    print("Segmentation-based spatial loss will not be available.")

logger = logging.getLogger(__name__)

class CellSegmenter:
    """Cell segmentation using Cellpose"""
    def __init__(self, model_type='cyto2', device='cuda', diameter=None):
        if not CELLPOSE_AVAILABLE:
            raise ImportError("cellpose is required. Install with: pip install cellpose")
        
        # Force GPU usage for cellpose
        use_gpu = (device == 'cuda' or device.startswith('cuda'))
        
        # Check if GPU is actually available
        if use_gpu and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU for cellpose")
            use_gpu = False
        
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        self.diameter = diameter
        
        if use_gpu:
            logger.info("Cellpose initialized with GPU support")
        else:
            logger.info("Cellpose initialized with CPU")
        
    def segment(self, images: np.ndarray, channels=None) -> List[np.ndarray]:
        """
        Segment cells from images
        
        Args:
            images: numpy array of shape [B, H, W, C] or [B, C, H, W]
            channels: [[cytoplasm, nucleus]] for cellpose. Default [[0, 0]] for grayscale
            
        Returns:
            List of masks, one per image in batch
        """
        if channels is None:
            channels = [[0, 0]]  # Grayscale
            
        masks_list = []
        for img in images:
            # Convert to [H, W, C] if needed
            if img.shape[0] <= 4:  # Likely [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            
            # Use first channel or RGB for segmentation
            if img.shape[2] >= 3:
                seg_img = img[:, :, :3]
            else:
                seg_img = img[:, :, 0]
                
            masks, _, _, _ = self.model.eval(seg_img, diameter=self.diameter, channels=channels)
            masks_list.append(masks)
            
        return masks_list


def compute_centroids(mask: np.ndarray) -> np.ndarray:
    """
    Compute centroids of segmented cells
    
    Args:
        mask: segmentation mask where each cell has a unique integer label
        
    Returns:
        Array of shape [N, 2] with centroid coordinates (y, x)
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background
    
    centroids = []
    for label in unique_labels:
        coords = np.argwhere(mask == label)
        centroid = coords.mean(axis=0)
        centroids.append(centroid)
        
    return np.array(centroids) if centroids else np.zeros((0, 2))


def build_distance_graph(centroids: np.ndarray, k_neighbors: int = 5, 
                         max_distance: Optional[float] = None) -> np.ndarray:
    """
    Build a distance-weighted graph from cell centroids
    
    Args:
        centroids: Array of shape [N, 2] with centroid coordinates
        k_neighbors: Number of nearest neighbors to connect
        max_distance: Maximum distance for connections (None = no limit)
        
    Returns:
        Adjacency matrix of shape [N, N] with distance weights
    """
    n = len(centroids)
    if n == 0:
        return np.zeros((0, 0))
    
    # Compute pairwise distances
    distances = cdist(centroids, centroids, metric='euclidean')
    
    # Build adjacency matrix with k-nearest neighbors
    adjacency = np.zeros_like(distances)
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances[i])[1:k_neighbors+1]
        
        for j in nearest_indices:
            dist = distances[i, j]
            if max_distance is None or dist <= max_distance:
                adjacency[i, j] = dist
                adjacency[j, i] = dist  # Symmetric
                
    return adjacency


def wasserstein_distance_graphs(centroids1: np.ndarray, centroids2: np.ndarray) -> float:
    """
    Compute approximate Wasserstein distance between two sets of centroids
    Uses optimal transport with uniform weights
    
    Args:
        centroids1: First set of centroids [N1, 2]
        centroids2: Second set of centroids [N2, 2]
        
    Returns:
        Wasserstein distance (scalar)
    """
    n1, n2 = len(centroids1), len(centroids2)
    
    if n1 == 0 or n2 == 0:
        # Handle empty graphs - return large distance
        return float(max(n1, n2) * 100)
    
    # Compute cost matrix (pairwise distances)
    cost_matrix = cdist(centroids1, centroids2, metric='euclidean')
    
    # Use Hungarian algorithm for optimal assignment
    # Pad to square matrix if needed
    max_n = max(n1, n2)
    padded_cost = np.full((max_n, max_n), cost_matrix.max() * 2)
    padded_cost[:n1, :n2] = cost_matrix
    
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Compute total cost
    total_cost = 0.0
    matched = 0
    for i, j in zip(row_ind, col_ind):
        if i < n1 and j < n2:
            total_cost += cost_matrix[i, j]
            matched += 1
    
    # Add penalty for unmatched nodes
    unmatched = abs(n1 - n2)
    if unmatched > 0:
        avg_cost = total_cost / max(matched, 1)
        total_cost += unmatched * avg_cost * 1.5
    
    return total_cost / max_n  # Normalize by size


def graph_structure_distance(adj1: np.ndarray, adj2: np.ndarray, 
                             centroids1: np.ndarray, centroids2: np.ndarray) -> float:
    """
    Compute distance between graph structures based on edge distributions
    
    Args:
        adj1, adj2: Adjacency matrices
        centroids1, centroids2: Centroid coordinates for alignment
        
    Returns:
        Structure distance (scalar)
    """
    if len(adj1) == 0 or len(adj2) == 0:
        return abs(len(adj1) - len(adj2)) * 10.0
    
    # Extract edge distances from adjacency matrices
    edges1 = adj1[adj1 > 0]
    edges2 = adj2[adj2 > 0]
    
    if len(edges1) == 0 or len(edges2) == 0:
        return abs(len(edges1) - len(edges2)) * 5.0
    
    # Compare edge distance distributions using Wasserstein
    edges1_sorted = np.sort(edges1)
    edges2_sorted = np.sort(edges2)
    
    # Interpolate to same length for comparison
    n_samples = 100
    edges1_interp = np.interp(np.linspace(0, 1, n_samples), 
                              np.linspace(0, 1, len(edges1_sorted)), 
                              edges1_sorted)
    edges2_interp = np.interp(np.linspace(0, 1, n_samples), 
                              np.linspace(0, 1, len(edges2_sorted)), 
                              edges2_sorted)
    
    distance = np.mean(np.abs(edges1_interp - edges2_interp))
    
    # Add penalty for different graph sizes
    size_penalty = abs(len(adj1) - len(adj2)) * 0.5
    
    return distance + size_penalty


def compute_segmentation_spatial_loss(generated_images: torch.Tensor, 
                                      truth_images: torch.Tensor,
                                      segmenter: CellSegmenter,
                                      k_neighbors: int = 5,
                                      device: str = 'cuda') -> torch.Tensor:
    """
    Compute spatial graph loss based on cell segmentation
    
    Args:
        generated_images: Generated images [B, C, H, W]
        truth_images: Ground truth images [B, C, H, W]
        segmenter: CellSegmenter instance
        k_neighbors: Number of neighbors for graph construction
        device: Device for computation
        
    Returns:
        Scalar loss tensor
    """
    batch_size = generated_images.shape[0]
    
    # Convert to numpy for segmentation
    gen_np = generated_images.detach().cpu().numpy()
    truth_np = truth_images.detach().cpu().numpy()
    
    total_loss = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        try:
            # Segment cells in both images
            gen_mask = segmenter.segment([gen_np[i]])[0]
            truth_mask = segmenter.segment([truth_np[i]])[0]
            
            # Compute centroids
            gen_centroids = compute_centroids(gen_mask)
            truth_centroids = compute_centroids(truth_mask)
            
            if len(gen_centroids) == 0 and len(truth_centroids) == 0:
                continue  # Both empty, skip
            
            # Build graphs
            gen_graph = build_distance_graph(gen_centroids, k_neighbors=k_neighbors)
            truth_graph = build_distance_graph(truth_centroids, k_neighbors=k_neighbors)
            
            # Compute graph distances
            centroid_dist = wasserstein_distance_graphs(gen_centroids, truth_centroids)
            structure_dist = graph_structure_distance(gen_graph, truth_graph, 
                                                     gen_centroids, truth_centroids)
            
            # Combine distances
            sample_loss = centroid_dist + 0.5 * structure_dist
            total_loss += sample_loss
            valid_samples += 1
            
        except Exception as e:
            logger.warning(f"Error computing segmentation spatial loss for sample {i}: {e}")
            continue
    
    if valid_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    avg_loss = total_loss / valid_samples
    return torch.tensor(avg_loss, device=device, dtype=torch.float32, requires_grad=False)

class SimpleSpatialLoss(nn.Module):
    """
    Fast spatial loss without segmentation
    Combines gradients, texture, and multi-scale information
    """
    def __init__(self, gradient_weight=1.0, texture_weight=0.5, device='cuda'):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.texture_weight = texture_weight
        self.device = device
        
        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.to(device)
    
    def compute_gradients(self, img):
        """Compute image gradients"""
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img[:, 0:1]
        
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return grad_mag
    
    def compute_texture(self, img, patch_size=16):
        """Compute local texture statistics"""
        patches = F.unfold(img, kernel_size=patch_size, stride=patch_size//2)
        patches = patches.view(img.shape[0], img.shape[1], patch_size*patch_size, -1)
        
        mean = patches.mean(dim=2)
        var = patches.var(dim=2)
        
        return mean, var
    
    def forward(self, generated, target):
        """
        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]
        Returns:
            Scalar loss tensor
        """
        assert generated.dtype == torch.float32, f"Expected float32, got {generated.dtype}"
        assert target.dtype == torch.float32, f"Expected float32, got {target.dtype}"
    
        total_loss = 0
        
        # 1. Multi-scale gradient loss
        for scale in [1, 2, 4]:
            if scale > 1:
                gen_scaled = F.avg_pool2d(generated, scale)
                target_scaled = F.avg_pool2d(target, scale)
            else:
                gen_scaled = generated
                target_scaled = target
            
            gen_grad = self.compute_gradients(gen_scaled)
            target_grad = self.compute_gradients(target_scaled)
            
            grad_loss = F.l1_loss(gen_grad, target_grad)
            total_loss += self.gradient_weight * grad_loss / scale
        
        # 2. Texture statistics loss
        gen_mean, gen_var = self.compute_texture(generated)
        target_mean, target_var = self.compute_texture(target)
        
        texture_loss = F.l1_loss(gen_mean, target_mean) + 0.5 * F.l1_loss(gen_var, target_var)
        total_loss += self.texture_weight * texture_loss
        
        return total_loss

class SpatialGraphLossModule:
    """
    Module for computing spatial loss with warmup
    Supports both segmentation-based and simple gradient-based methods
    """
    def __init__(self, 
                 method='simple',  # 'simple' or 'segmentation'
                 model_type='cyto2', 
                 device='cuda', 
                 k_neighbors=5, 
                 warmup_epochs=0, 
                 start_epoch=0,
                 gradient_weight=1.0,
                 texture_weight=0.5):
        """
        Args:
            method: 'simple' for gradient-based or 'segmentation' for cellpose-based
            model_type: Cellpose model type (only for segmentation method)
            device: Device for computation
            k_neighbors: Number of neighbors for graph (only for segmentation method)
            warmup_epochs: Number of epochs to warmup loss weight
            start_epoch: Epoch to start applying loss
            gradient_weight: Weight for gradient loss (only for simple method)
            texture_weight: Weight for texture loss (only for simple method)
        """
        self.method = method
        self.k_neighbors = k_neighbors
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        
        if method == 'segmentation':
            if not CELLPOSE_AVAILABLE:
                logger.error("Segmentation method requires cellpose. Falling back to simple method.")
                self.method = 'simple'
            else:
                self.segmenter = CellSegmenter(model_type=model_type, device=device)
                logger.info(f"Spatial loss initialized with SEGMENTATION method (k_neighbors={k_neighbors})")
        
        if self.method == 'simple':
            self.loss_fn = SimpleSpatialLoss(
                gradient_weight=gradient_weight,
                texture_weight=texture_weight,
                device=device
            )
            logger.info(f"Spatial loss initialized with SIMPLE method (gradient_weight={gradient_weight}, texture_weight={texture_weight})")
        
    def get_loss_weight(self, epoch: int) -> float:
        """
        Compute warmup weight for spatial loss
        
        Args:
            epoch: Current epoch
            
        Returns:
            Weight multiplier (0.0 to 1.0)
        """
        self.current_epoch = epoch
        
        # Before start_epoch, return 0
        if epoch < self.start_epoch:
            return 0.0
        
        # During warmup, linearly increase from 0 to 1
        if self.warmup_epochs > 0:
            warmup_progress = (epoch - self.start_epoch + 1) / self.warmup_epochs
            return min(1.0, warmup_progress)
        
        # After warmup, return 1
        return 1.0
        
    def __call__(self, generated_images: torch.Tensor, 
                 truth_images: torch.Tensor,
                 epoch: int) -> Tuple[torch.Tensor, float]:
        """
        Compute spatial loss with warmup
        
        Args:
            generated_images: Generated images [B, C, H, W]
            truth_images: Ground truth images [B, C, H, W]
            epoch: Current epoch
        
        Returns:
            Tuple of (loss_tensor, weight_used)
        """
        weight = self.get_loss_weight(epoch)
        
        if weight == 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=False), 0.0
        
        if self.method == 'segmentation':
            loss = compute_segmentation_spatial_loss(
                generated_images, truth_images, 
                self.segmenter, self.k_neighbors, self.device
            )
        else:  # simple method
            loss = self.loss_fn(generated_images, truth_images)
        
        return loss, weight

