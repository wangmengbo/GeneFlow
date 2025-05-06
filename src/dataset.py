import json
import torch
import logging
import tifffile
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================
# Dataset Implementation
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

class CellImageGeneDataset(Dataset):
    """Dataset for cell images and gene expression profiles with improved preprocessing"""
    def __init__(self, expr_df, image_paths, img_size=256, img_channels=3, 
                 transform=None, missing_gene_symbols=None, normalize_aux=False):
        """
        Args:
            expr_df: DataFrame with gene expression data (genes as rows, cells as columns)
            image_paths: JSON file mapping cell IDs to image paths or image paths dictionary
            img_size: Size to resize images to
            transform: Optional transforms to apply to images
        """
        self.expr_df = expr_df
        self.gene_list = expr_df.columns.tolist()
        self.normalize_aux = normalize_aux

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
        # If there are known missing genes, set corresponding mask indices to 0
        if self.missing_gene_indices:
            indices_to_zero = list(self.missing_gene_indices.values())
            if indices_to_zero: # Ensure list is not empty
                gene_mask[indices_to_zero] = 0
        
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
                if image.dtype != np.uint8:
                    # Normalize to [0, 1]
                    image = normalize_rgb(image)
                    
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
            'gene_mask': gene_mask,
            'image': image
        }


def patch_collate_fn(batch):
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


class PatchImageGeneDataset(Dataset):
    """Dataset for patch-level images and corresponding cell gene expression profiles"""
    def __init__(self, expr_df, patch_image_paths, patch_to_cells, img_size=256, img_channels=3, 
                 transform=None, normalize_aux=False):
        """
        Args:
            expr_df: DataFrame with gene expression data (cells as index, genes as columns)
            patch_image_paths: Dict/JSON mapping patch IDs to image paths
            patch_to_cells: Dict/JSON mapping patch IDs to lists of cell IDs
            img_size: Size to resize images to
            img_channels: Number of image channels to use
            transform: Optional transforms to apply to images
        """
        self.expr_df = expr_df
        self.gene_list = expr_df.columns.tolist()
        self.img_size = img_size
        self.img_channels = img_channels
        self.normalize_aux = normalize_aux
        
        # Load patch image paths
        if isinstance(patch_image_paths, str):
            with open(patch_image_paths, 'r') as f:
                self.patch_image_paths = json.load(f)
        else:
            self.patch_image_paths = patch_image_paths
            
        # Load patch to cells mapping
        if isinstance(patch_to_cells, str):
            with open(patch_to_cells, 'r') as f:
                self.patch_to_cells = json.load(f)
        else:
            self.patch_to_cells = patch_to_cells
        
        # Validate patches - only keep patches that have both image paths and cells in expression data
        self.valid_patches = []
        all_cells = set(self.expr_df.index)
        
        for patch_id, cells in self.patch_to_cells.items():
            if (patch_id in self.patch_image_paths and
                all(cell in all_cells for cell in cells)):
                self.valid_patches.append(patch_id)
        
        logger.info(f"Dataset contains {len(self.valid_patches)} valid patches")
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
        
        # Validate num_cells
        num_cells = len(cell_ids)
        if gene_exprs.shape[0] != num_cells:
            logger.error(f"Patch {patch_id}: num_cells={num_cells}, but gene_exprs has {gene_exprs.shape[0]} cells")
            raise ValueError(f"Patch {patch_id}: num_cells={num_cells}, but gene_exprs has {gene_exprs.shape[0]} cells")
        
        # Load and preprocess the patch image
        img_path = self.patch_image_paths[patch_id]
        image = self._load_image(img_path)
        
        return {
            'patch_id': patch_id,
            'cell_ids': cell_ids,
            'gene_expr': gene_exprs,
            'image': image,
            'num_cells': num_cells
        }
        
    def _load_image(self, img_path):
        """Load and preprocess an image with support for multi-channel images"""
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
                if image.dtype != np.uint8:
                    image = normalize_rgb(image)
                    
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
        
        return image