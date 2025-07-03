import json
import torch
import logging
import tifffile
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import hest
from hest import load_hest
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# HEST Integration Functions
# ======================================

def load_hest_data(hest_id=None, hest_path=None, save_path='./hest_data'):
    """
    Load HEST data either by ID or from local path
    Args:
        hest_id: HEST dataset ID (e.g., 'TENX13')
        hest_path: Path to local HEST data
        save_path: Path to save downloaded HEST data (not used with local path)
    Returns:
        hest_data: Loaded HEST dataset object
    """
    if hest_path:
        logger.info(f"Loading HEST dataset from local path: {hest_path}")
        # For local data, you might need to use iter_hest or load specific files
        from hest import iter_hest
        import os
        
        # Check if this is a directory with HEST data
        if os.path.isdir(hest_path):
            # Get available dataset IDs from the local path
            # You might need to specify which dataset to load
            # For now, let's try to load the first available dataset
            try:
                # Try to iterate through available datasets
                for hest_data in iter_hest(hest_path, id_list=hest_id if type(hest_id) is list else [hest_id]):
                    logger.info(f"Loaded HEST dataset: {hest_data}")
                    return hest_data
            except Exception as e:
                logger.error(f"Error loading from local path: {e}")
                # Fallback: try to load a specific dataset ID from local data
                # You might need to specify which dataset ID you want to use
                available_ids = ['TENX13', 'TENX95', 'TENX96']  # Common IDs
                for dataset_id in available_ids:
                    try:
                        hest_data = load_hest(dataset_id)
                        logger.info(f"Loaded HEST dataset {dataset_id} from local cache")
                        return hest_data
                    except:
                        continue
                raise ValueError(f"Could not load any HEST dataset from {hest_path}")
        else:
            # Try to load as a specific file
            hest_data = load_hest(hest_path)
    elif hest_id:
        logger.info(f"Loading HEST dataset with ID: {hest_id}")
        # Remove save_path parameter as it's not supported
        hest_data = load_hest(hest_id)
    else:
        raise ValueError("Either hest_id or hest_path must be provided")
    
    return hest_data


def extract_hest_features(hest_data, patch_size=224):
    """
    Extract features and metadata from HEST data
    
    Args:
        hest_data: HEST dataset object
        patch_size: Size of image patches to extract
    
    Returns:
        dict: Contains expression data, spatial coordinates, and image patches
    """
    # Get expression data
    expr_df = hest_data.adata.to_df()
    
    # Get spatial coordinates
    spatial_coords = hest_data.adata.obsm['spatial']
    
    # Get image data
    img = hest_data.img
    
    # Create patches based on spatial coordinates
    patches = {}
    patch_coords = {}
    
    for i, (x, y) in enumerate(spatial_coords):
        spot_id = hest_data.adata.obs.index[i]
        
        # Extract patch around spatial coordinate
        x_start = max(0, int(x - patch_size // 2))
        y_start = max(0, int(y - patch_size // 2))
        x_end = min(img.shape[1], x_start + patch_size)
        y_end = min(img.shape[0], y_start + patch_size)
        
        patch = img[y_start:y_end, x_start:x_end]
        patches[spot_id] = patch
        patch_coords[spot_id] = (x, y)
    
    return {
        'expression': expr_df,
        'spatial_coords': patch_coords,
        'patches': patches,
        'full_image': img,
        'adata': hest_data.adata
    }

# ======================================
# Enhanced Dataset Implementation
# ======================================

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

class HESTXeniumDataset(Dataset):
    """Enhanced dataset that combines Xenium data with HEST functionality"""
    
    def __init__(self, expr_df=None, image_paths=None, img_size=256, img_channels=3, 
                 transform=None, missing_gene_symbols=None, normalize_aux=False,
                 hest_id=None, hest_path=None, hest_save_path='./hest_data',
                 use_hest_patches=False, spatial_coords=None):
        """
        Args:
            expr_df: DataFrame with gene expression data (genes as rows, cells as columns)
            image_paths: JSON file mapping cell IDs to image paths or image paths dictionary
            img_size: Size to resize images to
            img_channels: Number of image channels
            transform: Optional transforms to apply to images
            missing_gene_symbols: List of missing gene symbols
            normalize_aux: Whether to normalize auxiliary channels
            hest_id: HEST dataset ID to load
            hest_path: Path to local HEST data
            hest_save_path: Path to save HEST data
            use_hest_patches: Whether to use HEST-extracted patches
            spatial_coords: Spatial coordinates for cells/spots
        """
        self.img_size = img_size
        self.img_channels = img_channels
        self.normalize_aux = normalize_aux
        self.use_hest_patches = use_hest_patches
        self.spatial_coords = spatial_coords or {}
        
        # Initialize HEST data if provided
        self.hest_data = None
        self.hest_features = None
        
        if hest_id or hest_path:
            logger.info("Loading HEST data...")
            self.hest_data = load_hest_data(hest_id, hest_path, hest_save_path)
            self.hest_features = extract_hest_features(self.hest_data, patch_size=img_size)
            
            # Use HEST expression data if no Xenium data provided
            if expr_df is None:
                expr_df = self.hest_features['expression']
                logger.info("Using HEST expression data")
            
            # Use HEST patches if requested and no image paths provided
            if use_hest_patches and image_paths is None:
                self.image_patches = self.hest_features['patches']
                self.spatial_coords = self.hest_features['spatial_coords']
                logger.info("Using HEST image patches")
        
        # Set up expression data
        if expr_df is not None:
            self.expr_df = expr_df
            self.gene_list = expr_df.columns.tolist()
        else:
            raise ValueError("No expression data provided (either expr_df or HEST data required)")
        
        # Set up image data
        if image_paths is not None:
            # Load Xenium image paths
            if isinstance(image_paths, str):
                with open(image_paths, 'r') as f:
                    self.image_paths = json.load(f)
            elif isinstance(image_paths, dict):
                self.image_paths = image_paths
            
            # Filter to only include cells that have both expression data and images
            common_cells = set(self.expr_df.index) & set(self.image_paths.keys())
            self.cell_ids = list(common_cells)
            self.use_xenium_images = True
            
        elif hasattr(self, 'image_patches'):
            # Use HEST patches
            common_cells = set(self.expr_df.index) & set(self.image_patches.keys())
            self.cell_ids = list(common_cells)
            self.use_xenium_images = False
            
        else:
            raise ValueError("No image data provided (either image_paths or HEST data required)")
        
        logger.info(f"Dataset contains {len(self.cell_ids)} cells with both expression data and images")
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ])
        else:
            self.transform = transform
        
        # Handle missing genes
        self.missing_gene_symbols = missing_gene_symbols
        self.missing_gene_indices = None
        if self.missing_gene_symbols:
            self.missing_gene_indices = {gene: idx for idx, gene in enumerate(self.gene_list) 
                                        if gene in self.missing_gene_symbols}
            
        if self.missing_gene_indices:
            logger.info(f"Initialized dataset with {len(self.missing_gene_indices)} missing gene indices identified.")
        else:
            logger.info("Initialized dataset with no missing gene symbols provided or found in data.")
                        
    def __len__(self):
        return len(self.cell_ids)
    
    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        
        # Get gene expression data
        gene_expr = self.expr_df.loc[cell_id].values.astype(np.float32)

        # Handle missing genes
        gene_mask = np.ones_like(gene_expr)
        if self.missing_gene_indices:
            indices_to_zero = list(self.missing_gene_indices.values())
            if indices_to_zero:
                gene_mask[indices_to_zero] = 0
        
        # Load and preprocess image
        if self.use_xenium_images:
            img_path = self.image_paths[cell_id]
            image = self._load_xenium_image(img_path)
        else:
            # Use HEST patch
            image_patch = self.image_patches[cell_id]
            image = self._process_hest_patch(image_patch)
        
        # Get spatial coordinates if available
        spatial_coord = self.spatial_coords.get(cell_id, (0, 0))
        
        return {
            'cell_id': cell_id,
            'gene_expr': gene_expr,
            'gene_mask': gene_mask,
            'image': image,
            'spatial_coord': spatial_coord
        }
    
    def _process_hest_patch(self, patch):
        """Process HEST image patch"""
        try:
            # Ensure patch has the right number of channels
            if len(patch.shape) == 2:
                # Grayscale - convert to RGB if needed
                if self.img_channels == 3:
                    patch = np.stack([patch] * 3, axis=-1)
            elif len(patch.shape) == 3:
                # Multi-channel - take only the required channels
                patch = patch[:, :, :self.img_channels]
            
            # Normalize if needed
            if patch.dtype != np.uint8:
                patch = normalize_rgb(patch)
            
            # Convert to PIL for transforms
            if len(patch.shape) == 2:
                pil_img = PILImage.fromarray(patch, mode='L')
            else:
                pil_img = PILImage.fromarray(patch)
            
            # Apply transforms
            if self.transform:
                image = self.transform(pil_img)
            else:
                image = transforms.ToTensor()(pil_img)
                
            return image
            
        except Exception as e:
            logger.error(f"Error processing HEST patch: {e}")
            # Create blank image as fallback
            if self.img_channels == 1:
                pil_img = PILImage.new('L', (self.img_size, self.img_size), 0)
            else:
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
            if self.transform:
                return self.transform(pil_img)
            else:
                return transforms.ToTensor()(pil_img)
    
    def _load_xenium_image(self, img_path):
        """Load Xenium image (original functionality)"""
        try:
            # Try to open as TIFF first
            image = tifffile.imread(img_path)
            image = image[:,:,:self.img_channels]
            
            # Check if we have a multi-channel image with 3 or more channels
            if len(image.shape) == 3 and image.shape[2] >= 3:
                # Split into RGB (first 3 channels) and auxiliary channels (remaining channels)
                rgb_image = image[:, :, :3]
                
                # Get auxiliary channels if any exist
                aux_channels = []
                if image.shape[2] > 3:
                    for i in range(3, image.shape[2]):
                        aux_channels.append(image[:, :, i])
                
                # Normalize RGB if needed
                if rgb_image.dtype != np.uint8:
                    rgb_image = normalize_rgb(rgb_image)
                
                # Convert RGB to PIL image for transforms
                rgb_pil = PILImage.fromarray(rgb_image)
                
                # Process auxiliary channels
                aux_pil_channels = []
                for aux_channel in aux_channels:
                    # Normalize if needed
                    if self.normalize_aux and aux_channel.dtype != np.uint8:
                        aux_channel = normalize_aux(aux_channel)
                    
                    # Convert to PIL image and convert to RGB to match the expected channel count
                    aux_pil = PILImage.fromarray(aux_channel, mode='L')
                    aux_pil = aux_pil.convert('RGB')
                    aux_pil_channels.append(aux_pil)
                
                # Apply transforms
                if self.transform:
                    rgb_transformed = self.transform(rgb_pil)
                    
                    aux_transformed = []
                    for aux_pil in aux_pil_channels:
                        aux_transformed.append(self.transform(aux_pil))
                    
                    # Extract only the first channel from each aux tensor
                    aux_single_channels = []
                    for aux_tensor in aux_transformed:
                        aux_single_channel = aux_tensor[0:1]  # Shape: [1, H, W]
                        aux_single_channels.append(aux_single_channel)
                    
                    # Concatenate all tensors along the channel dimension
                    image = torch.cat([rgb_transformed] + aux_single_channels, dim=0)
                    
                else:
                    # If no transforms, convert to tensors manually
                    rgb_tensor = transforms.ToTensor()(rgb_pil)
                    
                    aux_tensors = []
                    for aux_pil in aux_pil_channels:
                        if aux_pil.mode == 'RGB':
                            aux_pil = aux_pil.convert('L')
                        aux_tensor = transforms.ToTensor()(aux_pil)
                        aux_tensors.append(aux_tensor)
                    
                    # Concatenate all tensors
                    image = torch.cat([rgb_tensor] + aux_tensors, dim=0)
                    
            else:
                # Handle standard images (1 or 3 channels)
                if image.dtype != np.uint8:
                    image = normalize_rgb(image)
                    
                # Convert to PIL image for transforms
                if len(image.shape) == 2:
                    pil_img = PILImage.fromarray(image, mode='L')
                    if self.img_channels == 3:
                        pil_img = pil_img.convert('RGB')
                else:
                    pil_img = PILImage.fromarray(image)
                    
                # Apply transforms
                if self.transform:
                    image = self.transform(pil_img)
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            if hasattr(self, 'img_channels') and self.img_channels == 1:
                pil_img = PILImage.new('L', (self.img_size, self.img_size), 0)
            else:
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
            # Apply transforms
            if self.transform:
                image = self.transform(pil_img)
        
        return image

    def get_hest_metadata(self):
        """Get HEST metadata if available"""
        if self.hest_data:
            return {
                'adata': self.hest_features['adata'],
                'full_image': self.hest_features['full_image'],
                'spatial_coords': self.hest_features['spatial_coords']
            }
        return None

# ======================================
# Enhanced Patch Dataset with HEST
# ======================================

class HESTPatchImageGeneDataset(Dataset):
    """Enhanced patch dataset that integrates HEST functionality"""
    
    def __init__(self, expr_df=None, patch_image_paths=None, patch_to_cells=None, 
                 img_size=256, img_channels=3, transform=None, normalize_aux=False,
                 hest_id=None, hest_path=None, hest_save_path='./hest_data'):
        """
        Args:
            expr_df: DataFrame with gene expression data
            patch_image_paths: Dict/JSON mapping patch IDs to image paths
            patch_to_cells: Dict/JSON mapping patch IDs to lists of cell IDs
            img_size: Size to resize images to
            img_channels: Number of image channels to use
            transform: Optional transforms to apply to images
            normalize_aux: Whether to normalize auxiliary channels
            hest_id: HEST dataset ID to load
            hest_path: Path to local HEST data
            hest_save_path: Path to save HEST data
        """
        self.img_size = img_size
        self.img_channels = img_channels
        self.normalize_aux = normalize_aux
        
        # Initialize HEST data if provided
        self.hest_data = None
        self.hest_features = None
        
        if hest_id or hest_path:
            logger.info("Loading HEST data for patch dataset...")
            self.hest_data = load_hest_data(hest_id, hest_path, hest_save_path)
            self.hest_features = extract_hest_features(self.hest_data, patch_size=img_size)
            
            # Use HEST data if no Xenium data provided
            if expr_df is None:
                expr_df = self.hest_features['expression']
                logger.info("Using HEST expression data for patches")
        
        if expr_df is not None:
            self.expr_df = expr_df
            self.gene_list = expr_df.columns.tolist()
        else:
            raise ValueError("No expression data provided")
        
        # Set up patch data
        if patch_image_paths is not None and patch_to_cells is not None:
            # Load Xenium patch data
            if isinstance(patch_image_paths, str):
                with open(patch_image_paths, 'r') as f:
                    self.patch_image_paths = json.load(f)
            else:
                self.patch_image_paths = patch_image_paths
                
            if isinstance(patch_to_cells, str):
                with open(patch_to_cells, 'r') as f:
                    self.patch_to_cells = json.load(f)
            else:
                self.patch_to_cells = patch_to_cells
                
        elif self.hest_features:
            # Create patches from HEST data
            self.patch_image_paths = {}
            self.patch_to_cells = {}
            
            # Group cells into patches (you might want to customize this logic)
            for i, cell_id in enumerate(self.hest_features['expression'].index):
                patch_id = f"hest_patch_{i}"
                self.patch_image_paths[patch_id] = cell_id  # Use cell_id as reference to patch
                self.patch_to_cells[patch_id] = [cell_id]
                
        else:
            raise ValueError("No patch data provided")
        
        # Validate patches
        self.valid_patches = []
        all_cells = set(self.expr_df.index)
        
        for patch_id, cells in self.patch_to_cells.items():
            if (patch_id in self.patch_image_paths and
                all(cell in all_cells for cell in cells)):
                self.valid_patches.append(patch_id)
        
        logger.info(f"Patch dataset contains {len(self.valid_patches)} valid patches")
        logger.info(f"Total number of cells across all patches: {sum(len(self.patch_to_cells[p]) for p in self.valid_patches)}")
        
        # Set up image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.valid_patches)
    
    def __getitem__(self, idx):
        patch_id = self.valid_patches[idx]
        cell_ids = self.patch_to_cells[patch_id]
        
        # Get gene expression data for all cells in this patch
        gene_exprs = torch.stack([
            torch.tensor(self.expr_df.loc[cell_id].values, dtype=torch.float32)
            for cell_id in cell_ids
        ])
        
        num_cells = len(cell_ids)
        if gene_exprs.shape[0] != num_cells:
            logger.error(f"Patch {patch_id}: num_cells={num_cells}, but gene_exprs has {gene_exprs.shape[0]} cells")
            raise ValueError(f"Patch {patch_id}: num_cells={num_cells}, but gene_exprs has {gene_exprs.shape[0]} cells")
        
        # Load and preprocess the patch image
        if self.hest_features and patch_id.startswith('hest_patch_'):
            # Use HEST patch
            cell_id = self.patch_image_paths[patch_id]
            image_patch = self.hest_features['patches'][cell_id]
            image = self._process_hest_patch(image_patch)
        else:
            # Use Xenium patch
            img_path = self.patch_image_paths[patch_id]
            image = self._load_image(img_path)
        
        return {
            'patch_id': patch_id,
            'cell_ids': cell_ids,
            'gene_expr': gene_exprs,
            'image': image,
            'num_cells': num_cells
        }
    
    def _process_hest_patch(self, patch):
        """Process HEST image patch"""
        try:
            # Ensure patch has the right number of channels
            if len(patch.shape) == 2:
                if self.img_channels == 3:
                    patch = np.stack([patch] * 3, axis=-1)
            elif len(patch.shape) == 3:
                patch = patch[:, :, :self.img_channels]
            
            # Normalize if needed
            if patch.dtype != np.uint8:
                patch = normalize_rgb(patch)
            
            # Convert to PIL for transforms
            if len(patch.shape) == 2:
                pil_img = PILImage.fromarray(patch, mode='L')
            else:
                pil_img = PILImage.fromarray(patch)
            
            # Apply transforms
            if self.transform:
                image = self.transform(pil_img)
            else:
                image = transforms.ToTensor()(pil_img)
                
            return image
            
        except Exception as e:
            logger.error(f"Error processing HEST patch: {e}")
            # Create blank image as fallback
            if self.img_channels == 1:
                pil_img = PILImage.new('L', (self.img_size, self.img_size), 0)
            else:
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
            if self.transform:
                return self.transform(pil_img)
            else:
                return transforms.ToTensor()(pil_img)
        
    def _load_image(self, img_path):
        """Load and preprocess an image with support for multi-channel images (original Xenium functionality)"""
        try:
            # Try to open as TIFF first
            image = tifffile.imread(img_path)
            image = image[:,:,:self.img_channels]
            
            # Check if we have a multi-channel image with 3 or more channels
            if len(image.shape) == 3 and image.shape[2] >= 3:
                # Split into RGB (first 3 channels) and auxiliary channels (remaining channels)
                rgb_image = image[:, :, :3]
                
                # Get auxiliary channels if any exist
                aux_channels = []
                if image.shape[2] > 3:
                    for i in range(3, image.shape[2]):
                        aux_channels.append(image[:, :, i])
                
                # Normalize RGB if needed
                if rgb_image.dtype != np.uint8:
                    rgb_image = normalize_rgb(rgb_image)
                
                # Convert RGB to PIL image for transforms
                rgb_pil = PILImage.fromarray(rgb_image)
                
                # Process auxiliary channels
                aux_pil_channels = []
                for aux_channel in aux_channels:
                    # Normalize if needed
                    if self.normalize_aux and aux_channel.dtype != np.uint8:
                        aux_channel = normalize_aux(aux_channel)
                    
                    # Convert to PIL image and convert to RGB to match the expected channel count
                    aux_pil = PILImage.fromarray(aux_channel, mode='L')
                    aux_pil = aux_pil.convert('RGB')  # Convert to RGB to match transform expectations
                    aux_pil_channels.append(aux_pil)
                
                # Apply transforms
                if self.transform:
                    rgb_transformed = self.transform(rgb_pil)
                    
                    aux_transformed = []
                    for aux_pil in aux_pil_channels:
                        aux_transformed.append(self.transform(aux_pil))
                    
                    # Extract only the first channel from each aux tensor
                    aux_single_channels = []
                    for aux_tensor in aux_transformed:
                        aux_single_channel = aux_tensor[0:1]  # Shape: [1, H, W]
                        aux_single_channels.append(aux_single_channel)
                    
                    # Concatenate all tensors along the channel dimension
                    image = torch.cat([rgb_transformed] + aux_single_channels, dim=0)
                    
                else:
                    # If no transforms, convert to tensors manually
                    rgb_tensor = transforms.ToTensor()(rgb_pil)
                    
                    aux_tensors = []
                    for aux_pil in aux_pil_channels:
                        if aux_pil.mode == 'RGB':
                            aux_pil = aux_pil.convert('L')
                        aux_tensor = transforms.ToTensor()(aux_pil)  # Shape: [1, H, W]
                        aux_tensors.append(aux_tensor)
                    
                    # Concatenate all tensors
                    image = torch.cat([rgb_tensor] + aux_tensors, dim=0)
                    
            else:
                # Handle standard images (1 or 3 channels)
                if image.dtype != np.uint8:
                    image = normalize_rgb(image)
                    
                # Convert to PIL image for transforms
                if len(image.shape) == 2:
                    pil_img = PILImage.fromarray(image, mode='L')
                    if self.img_channels == 3:
                        pil_img = pil_img.convert('RGB')
                else:
                    pil_img = PILImage.fromarray(image)
                    
                # Apply transforms
                if self.transform:
                    image = self.transform(pil_img)
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            if hasattr(self, 'img_channels') and self.img_channels == 1:
                pil_img = PILImage.new('L', (self.img_size, self.img_size), 0)
            else:
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
            # Apply transforms
            if self.transform:
                image = self.transform(pil_img)
        
        return image

# ======================================
# Enhanced Collate Function
# ======================================

def enhanced_collate_fn(batch):
    """Enhanced collate function that handles spatial coordinates"""
    cell_ids = [item['cell_id'] for item in batch]
    gene_exprs = torch.stack([torch.tensor(item['gene_expr']) for item in batch])
    gene_masks = torch.stack([torch.tensor(item['gene_mask']) for item in batch])
    images = torch.stack([item['image'] for item in batch])
    
    # Handle spatial coordinates if present
    spatial_coords = None
    if 'spatial_coord' in batch[0]:
        spatial_coords = torch.tensor([item['spatial_coord'] for item in batch])
    
    result = {
        'cell_id': cell_ids,
        'gene_expr': gene_exprs,
        'gene_mask': gene_masks,
        'image': images
    }
    
    if spatial_coords is not None:
        result['spatial_coord'] = spatial_coords
    
    return result

def patch_collate_fn(batch):
    """Original patch collate function"""
    patch_ids = [item['patch_id'] for item in batch]
    cell_ids_list = [item['cell_ids'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    num_cells = [item['num_cells'] for item in batch]
    
    gene_exprs = [item['gene_expr'] for item in batch]
    gene_dim = gene_exprs[0].shape[1]
    
    # Validate num_cells
    for i, (expr, n_cells) in enumerate(zip(gene_exprs, num_cells)):
        if expr.shape[0] != n_cells:
            logger.error(f"Sample {i} (patch {patch_ids[i]}): num_cells={n_cells}, but gene_expr has {expr.shape[0]} cells")
            raise ValueError(f"Sample {i} (patch {patch_ids[i]}): num_cells={n_cells}, but gene_expr has {expr.shape[0]} cells")
    
    max_cells = max(num_cells)
    batch_size = len(batch)
    padded_gene_exprs = torch.zeros(batch_size, max_cells, gene_dim, dtype=gene_exprs[0].dtype)
    
    for i, expr in enumerate(gene_exprs):
        n_cells = num_cells[i]
        padded_gene_exprs[i, :n_cells] = expr
    
    return {
        'patch_id': patch_ids,
        'cell_ids': cell_ids_list,
        'gene_expr': padded_gene_exprs,
        'image': images,
        'num_cells': torch.tensor(num_cells)
    }

# ======================================
# Usage Examples
# ======================================

def create_xenium_hest_dataset(xenium_expr_df, xenium_image_paths, hest_id=None):
    """
    Example function to create a dataset combining Xenium and HEST data
    """
    dataset = HESTXeniumDataset(
        expr_df=xenium_expr_df,
        image_paths=xenium_image_paths,
        hest_id=hest_id,
        img_size=256,
        img_channels=3,
        use_hest_patches=False  # Use Xenium images primarily
    )
    return dataset

def create_hest_only_dataset(hest_id):
    """
    Example function to create a dataset using only HEST data
    """
    dataset = HESTXeniumDataset(
        hest_id=hest_id,
        img_size=256,
        img_channels=3,
        use_hest_patches=True
    )
    return dataset
