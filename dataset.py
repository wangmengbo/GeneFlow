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