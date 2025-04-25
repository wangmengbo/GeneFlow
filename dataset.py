import json
import tifffile
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image as PILImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# Dataset Implementation
# ======================================

class CellImageGeneDataset(Dataset):
    """Dataset for cell images and gene expression profiles with improved preprocessing"""
    def __init__(self, expr_df, image_paths, img_size=256, img_channels=3, transform=None):
        """
        Args:
            expr_df: DataFrame with gene expression data (genes as rows, cells as columns)
            image_paths: JSON file mapping cell IDs to image paths or image paths dictionary
            img_size: Size to resize images to
            transform: Optional transforms to apply to images
        """
        self.expr_df = expr_df
        
        # Load image paths
        if isinstance(image_paths, str):
            with open(image_paths, 'r') as f:
                self.image_paths = json.load(f)
        elif isinstance(image_paths, dict):
            self.image_paths = image_paths
        
        # Filter to only include cells that have both expression data and images
        common_cells = set(self.expr_df.index) & set(self.image_paths.keys())
        self.cell_ids = list(common_cells)
        
        logger.info(f"Dataset contains {len(self.cell_ids)} cells with both expression data and images")
        
        self.img_size = img_size
        self.img_channels = img_channels
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
                # transforms.Normalize((0.5,), (0.5,)) if img_size == 1 else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.cell_ids)
    
    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        
        # Get gene expression data
        gene_expr = self.expr_df.loc[cell_id].values.astype(np.float32)
        
        # Load and preprocess image
        img_path = self.image_paths[cell_id]
        
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
                if rgb_image.dtype == np.uint16:
                    rgb_image = ((rgb_image - np.min(rgb_image) + 0.0001) / (np.max(rgb_image) - np.min(rgb_image) + 0.0001))
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                
                # Convert RGB to PIL image for transforms
                rgb_pil = PILImage.fromarray(rgb_image)
                
                # Process auxiliary channels
                aux_pil_channels = []
                for aux_channel in aux_channels:
                    # Normalize if needed
                    if aux_channel.dtype == np.uint16:
                        aux_channel = ((aux_channel - np.min(aux_channel) + 0.0001) / (np.max(aux_channel) - np.min(aux_channel) + 0.0001))
                        aux_channel = (aux_channel * 255).astype(np.uint8)
                    
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
                    
                    # Now we need to extract just the first channel from each aux_transformed
                    # since we converted them to RGB but only need the first channel
                    
                    # All transformed images should now be tensors with shape [C, H, W]
                    # For RGB: shape is [3, H, W]
                    # For aux (converted to RGB): shape is [3, H, W] but all channels are identical
                    
                    # Extract only the first channel from each aux tensor and reshape
                    aux_single_channels = []
                    for aux_tensor in aux_transformed:
                        # Take only the first channel and keep dimensions
                        aux_single_channel = aux_tensor[0:1]  # Shape: [1, H, W]
                        aux_single_channels.append(aux_single_channel)
                    
                    # Concatenate all tensors along the channel dimension
                    # RGB tensor shape: [3, H, W]
                    # Each aux tensor shape: [1, H, W]
                    image = torch.cat([rgb_transformed] + aux_single_channels, dim=0)
                    
                else:
                    # If no transforms, convert to tensors manually
                    rgb_tensor = transforms.ToTensor()(rgb_pil)
                    
                    aux_tensors = []
                    for aux_pil in aux_pil_channels:
                        # Convert back to grayscale if we converted to RGB earlier
                        if aux_pil.mode == 'RGB':
                            aux_pil = aux_pil.convert('L')
                        aux_tensor = transforms.ToTensor()(aux_pil)  # Shape: [1, H, W]
                        aux_tensors.append(aux_tensor)
                    
                    # Concatenate all tensors
                    image = torch.cat([rgb_tensor] + aux_tensors, dim=0)
                    
            else:
                # Handle standard images (1 or 3 channels)
                # Normalize TIFF image if it's 16-bit
                if image.dtype == np.uint16:
                    # Normalize to [0, 1]
                    image = ((image - np.min(image) + 0.0001) / (np.max(image) - np.min(image) + 0.0001))
                    image = (image * 255).astype(np.uint8)
                    
                # Convert to PIL image for transforms
                if len(image.shape) == 2:
                    # Grayscale
                    pil_img = PILImage.fromarray(image, mode='L')
                    # Convert to RGB if needed
                    if self.img_channels == 3:
                        pil_img = pil_img.convert('RGB')
                else:
                    # Already RGB
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
        
        return {
            'cell_id': cell_id,
            'gene_expr': gene_expr,
            'image': image
        }


class CellAndPatchImageGeneDataset(Dataset):
    """Dataset for cell images and gene expression profiles with flexible image selection and multi-channel support"""
    def __init__(self, expr_df, image_paths=None, img_size=256, img_channels=None, 
                 transform=None, patch_image_paths=None, patch_to_cells=None,
                 cell_image_key="masked_image", patch_image_key="original_patch",
                 normalize_image=True):
        """
        Args:
            expr_df: DataFrame with gene expression data (cells as index, genes as columns)
            image_paths: Dict/JSON mapping cell IDs to nested dict of image paths
            patch_image_paths: Dict/JSON mapping patch IDs to nested dict of image paths
            patch_to_cells: Dict/JSON mapping patch IDs to lists of cell IDs
            img_size: Size to resize images to
            img_channels: Number of image channels to use (None = use all available channels)
            transform: Optional transforms to apply to images
            cell_image_key: Key to select which cell image type to use (e.g., "masked_image")
            patch_image_key: Key to select which patch image type to use (e.g., "original_patch")
            normalize_image: Whether to normalize image values
        """
        self.expr_df = expr_df
        self.img_size = img_size
        self.img_channels = img_channels  # Can be None to use all available channels
        self.cell_image_key = cell_image_key
        self.patch_image_key = patch_image_key
        self.normalize_image = normalize_image
        
        # Load cell image paths
        if image_paths is not None:
            if isinstance(image_paths, str):
                with open(image_paths, 'r') as f:
                    self.image_paths = json.load(f)
            else:
                self.image_paths = image_paths
        else:
            self.image_paths = {}
            
        # Load patch data if provided
        self.has_patches = False
        if patch_image_paths is not None and patch_to_cells is not None:
            self.has_patches = True
            # Load patch image paths
            if isinstance(patch_image_paths, str):
                with open(patch_image_paths, 'r') as f:
                    self.patch_image_paths = json.load(f)
            else:
                self.patch_image_paths = patch_image_paths
                
            # Load patch cell IDs
            self.patch_to_cells = patch_to_cells
            logger.debug("self.patch_to_cells:", self.patch_to_cells)

            # Validate patches
            self.valid_patches = []
            all_cells = set(self.expr_df.index)
            for patch_id, cells in self.patch_to_cells.items():
                logger.debug("patch_id:", patch_id, "cells:", cells)
                if (patch_id in self.patch_image_paths and 
                    self.patch_image_key in self.patch_image_paths[patch_id] and
                    all(c in all_cells for c in cells)):
                    self.valid_patches.append(patch_id)
            logger.info(f"Found {len(self.valid_patches)} valid patches with complete data")
        
        # Find valid single cells
        self.valid_cells = []
        if self.image_paths:
            for cell_id in self.expr_df.index:
                if (cell_id in self.image_paths and 
                    self.cell_image_key in self.image_paths[cell_id]):
                    self.valid_cells.append(cell_id)
                    
            logger.info(f"Found {len(self.valid_cells)} valid cells with complete data")
        
        # Combine all items (cells and patches)
        self.all_items = self.valid_cells + self.valid_patches
        
        # Initialize transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size), antialias=True),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.all_items)
    
    def __getitem__(self, idx):
        item_id = self.all_items[idx]
    
        # Check if this is a patch or a cell
        if self.has_patches and item_id in self.valid_patches:
            item = self._get_patch_item(item_id)
        else:
            item = self._get_cell_item(item_id)
        
        # Check for NaN values
        if torch.isnan(item['gene_expr']).any():
            logger.warning(f"NaN values found in gene expression for item {item_id}")
            # Replace NaNs with zeros
            item['gene_expr'] = torch.nan_to_num(item['gene_expr'], nan=0.0)
        
        if torch.isnan(item['image']).any():
            logger.warning(f"NaN values found in image for item {item_id}")
            # Replace NaNs with zeros
            item['image'] = torch.nan_to_num(item['image'], nan=0.0)
        
        return item
        
    def _get_cell_item(self, cell_id):
        """Get a single cell item"""
        gene_expr = torch.tensor(self.expr_df.loc[cell_id].values, dtype=torch.float32)
        img_path = self.image_paths[cell_id][self.cell_image_key]
        
        return {
            'id': cell_id,
            'gene_expr': gene_expr,
            'image': self._load_image(img_path),
            'num_cells': 1,
            'cell_ids': [cell_id],
            'is_patch': False
        }

    def _get_patch_item(self, patch_id):
        """Get a patch item with multiple cells"""
        cell_ids = self.patch_to_cells[patch_id]
    
        # Convert to tensor directly
        gene_exprs = torch.stack([
            torch.tensor(self.expr_df.loc[cid].values, dtype=torch.float32)
            for cid in cell_ids
        ])
        img_path = self.patch_image_paths[patch_id][self.patch_image_key]
        
        return {
            'id': patch_id,
            'gene_expr': gene_exprs,
            'image': self._load_image(img_path),
            'num_cells': len(cell_ids),
            'cell_ids': cell_ids,
            'is_patch': True
        }
        
    def _load_image(self, img_path):
        """Load and preprocess an image with support for multi-channel images"""
        try:
            # Try to open as TIFF
            image = tifffile.imread(img_path)
            
            # Handle channel-first format (C,H,W) -> (H,W,C)
            if len(image.shape) == 3 and image.shape[0] <= 10:  # Assuming max 10 channels for safety
                image = np.transpose(image, (1, 2, 0))
            
            # Remove singleton dimensions
            if len(image.shape) > 2 and 1 in image.shape:
                image = np.squeeze(image)
            
            # Process multi-channel images
            if len(image.shape) == 3:
                # Get number of channels in the image
                num_channels = image.shape[2]
                
                # Limit channels if specified
                if self.img_channels is not None and num_channels > self.img_channels:
                    image = image[:, :, :self.img_channels]
                    num_channels = self.img_channels
                
                # Check if we have more than 3 channels
                if num_channels > 3:
                    # Split into RGB (first 3 channels) and auxiliary channels
                    rgb_image = image[:, :, :3]
                    aux_channels = [image[:, :, i] for i in range(3, num_channels)]
                    
                    # Normalize RGB if needed
                    if self.normalize_image:
                        if rgb_image.dtype in [np.float32, np.float64]:
                            if np.min(rgb_image) < 0 or np.max(rgb_image) > 1.0:
                                rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image) + 1e-8)
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        elif rgb_image.dtype == np.uint16:
                            rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image) + 1e-8) * 255).astype(np.uint8)
                    
                    # Convert RGB to PIL image
                    rgb_pil = PILImage.fromarray(rgb_image)
                    
                    # Process auxiliary channels
                    aux_pil_channels = []
                    for aux_channel in aux_channels:
                        # Normalize if needed
                        if self.normalize_image:
                            if aux_channel.dtype in [np.float32, np.float64]:
                                if np.min(aux_channel) < 0 or np.max(aux_channel) > 1.0:
                                    aux_channel = (aux_channel - np.min(aux_channel)) / (np.max(aux_channel) - np.min(aux_channel) + 1e-8)
                                aux_channel = (aux_channel * 255).astype(np.uint8)
                            elif aux_channel.dtype == np.uint16:
                                aux_channel = ((aux_channel - np.min(aux_channel)) / (np.max(aux_channel) - np.min(aux_channel) + 1e-8) * 255).astype(np.uint8)
                        
                        # Convert to PIL image
                        aux_pil = PILImage.fromarray(aux_channel, mode='L')
                        aux_pil_channels.append(aux_pil)
                    
                    # Apply transforms
                    rgb_transformed = self.transform(rgb_pil)
                    
                    # Transform each auxiliary channel
                    aux_transformed = []
                    for aux_pil in aux_pil_channels:
                        aux_tensor = self.transform(aux_pil)
                        # Extract only the first channel (since ToTensor converts L to [1,H,W])
                        aux_transformed.append(aux_tensor[0:1])
                    
                    # Concatenate all tensors along the channel dimension
                    return torch.cat([rgb_transformed] + aux_transformed, dim=0)
                
                else:
                    # Standard 3-channel or fewer image
                    if self.normalize_image:
                        if image.dtype in [np.float32, np.float64]:
                            if np.min(image) < 0 or np.max(image) > 1.0:
                                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
                            image = (image * 255).astype(np.uint8)
                        elif image.dtype == np.uint16:
                            image = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)
                    
                    pil_img = PILImage.fromarray(image)
                    return self.transform(pil_img)
            
            elif len(image.shape) == 2:  # Grayscale
                # Normalize if needed
                if self.normalize_image:
                    if image.dtype in [np.float32, np.float64]:
                        if np.min(image) < 0 or np.max(image) > 1.0:
                            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
                        image = (image * 255).astype(np.uint8)
                    elif image.dtype == np.uint16:
                        image = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)
                
                pil_img = PILImage.fromarray(image, mode='L')
                return self.transform(pil_img)
            
            else:
                raise ValueError(f"Unsupported image dimensions: {image.shape}")
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            if self.img_channels is not None:
                channels = self.img_channels
            else:
                channels = 3  # Default to 3 channels
            
            # Create a tensor of the right shape with zeros
            blank_image = torch.zeros(channels, self.img_size, self.img_size)
            return blank_image

