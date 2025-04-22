import json
import tifffile
import numpy as np
import logging
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
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            if self.img_channels == 1:
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