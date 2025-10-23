import json
import torch
import logging
import tifffile
import os
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.utils import normalize_rgb, normalize_aux
from typing import List, Dict, Optional
import glob
import pickle
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CellImageGeneDataset(Dataset):
    """Dataset for cell images and gene expression profiles, supporting in-memory images."""

    def __init__(self, expr_df, image_paths, img_size=256, img_channels=3,
                 transform=None, missing_gene_symbols=None, normalize_aux=False):
        self.expr_df = expr_df
        self.gene_list = expr_df.columns.tolist()
        self.normalize_aux = normalize_aux

        # Allow image_paths to be a dict of file paths OR a dict of numpy arrays (pre-loaded patches)
        self.image_paths = image_paths

        # Filter to only include cells that have both expression data and images
        # print(self.expr_df.head())
        # print(list(self.image_paths.keys())[:10])
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
            logger.info(f"Initialized dataset with {len(self.missing_gene_indices)} missing gene indices identified.")
        else:
            logger.info("Initialized dataset with no missing gene symbols provided or found in data.")

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        gene_expr = self.expr_df.loc[cell_id].values.astype(np.float32)
        gene_mask = np.ones_like(gene_expr)
        if self.missing_gene_indices:
            indices_to_zero = list(self.missing_gene_indices.values())
            if indices_to_zero:
                gene_mask[indices_to_zero] = 0

        img_source = self.image_paths[cell_id]
        # If it's a numpy array, treat as in-memory patch
        if isinstance(img_source, np.ndarray):
            patch = img_source
            if patch.shape[-1] != self.img_channels:
                patch = patch[..., :self.img_channels]
            if patch.dtype != np.uint8:
                patch = normalize_rgb(patch)
            pil_img = PILImage.fromarray(patch)
            image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        else:
            # Otherwise, treat as file path (legacy)
            try:
                image = tifffile.imread(img_source)
                image = image[:, :, :self.img_channels]
                if image.dtype != np.uint8:
                    image = normalize_rgb(image)
                pil_img = PILImage.fromarray(image)
                image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
            except Exception as e:
                logger.error(f"Error loading image {img_source}: {e}")
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
                image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)

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


def load_preprocessed_hest1k_singlecell_data(sids, base_dir, img_size=224, img_channels=3, 
    filter_min_genes=10, filter_min_counts=0,
    normalize_total=True, log1p=True):
    # Check if sid is a iterable (e.g., list of sample IDs)
    if not isinstance(sids, list):
        sids = [sids]
    
    expr_df = []
    image_paths = {}
    for sid in sids:
        # Load gene expression
        p_st = os.path.join(base_dir, "st", f"{sid}.h5ad")
        st = sc.read_h5ad(p_st)
        if filter_min_genes > 0:
            sc.pp.filter_cells(st, min_genes=filter_min_genes)
        if filter_min_counts > 0:
            sc.pp.filter_cells(st, min_counts=filter_min_counts)
        if normalize_total:
            sc.pp.normalize_total(st, target_sum=1e6)
        if log1p:
            sc.pp.log1p(st)
        expr_df_tmp = st.to_df()
        expr_df_tmp.index = [ f"{sid}.{cell_id}" for cell_id in expr_df_tmp.index ]

        # Ensure gene expressiong data has no nans
        if expr_df_tmp.isnull().values.any():
            logger.warning(f"Gene expression data for {sid} contains NaNs. Filling with zeros.")
            expr_df_tmp.fillna(0, inplace=True)
        # print(expr_df.head())
        expr_df.append(expr_df_tmp)

        # Load patch images and barcodes
        p_patches = os.path.join(base_dir, "patches", f"{sid}.h5")
        with h5py.File(p_patches, 'r') as f:
            barcode = np.array(f['barcode']).astype(str).flatten()
            barcode = np.char.add(f"{sid}.", barcode)
            img = np.array(f['img'])  # shape: [N, H, W, C]
        # Build image_paths dict: cell_id -> patch image (numpy array)
        image_paths.update({cell_id: img[i] for i, cell_id in enumerate(barcode)})

        # Check if expr_df_tmp has any nan or inf value with pandas method
        print(f"Sample {sid}: expr_df_tmp nan:{expr_df_tmp.isnull().values.any()}, inf:{np.isinf(expr_df_tmp.values).any().any()}")
        print(f"img, nan:{np.isnan(img).any()}, inf:{np.isinf(img).any()}")

    # Check shapes of all expression data frames
    logger.info(f"Loaded {len(expr_df)} samples with shape {[expr_df_tmp.shape for expr_df_tmp in expr_df]}")

    expr_df = pd.concat(expr_df, axis=0, join='outer', sort=False)
    expr_df.fillna(0, inplace=True)  # Fill NaNs with zeros for consistency
    expr_df = expr_df.astype(np.float32)  # Ensure all data is float32 for consistency
    print(f"expr_df.head():\n{expr_df.head()}")
    logger.info(f"Combined expression data shape: {expr_df.shape}")
    return expr_df, image_paths


class OnDemandMultiSampleHestXeniumDataset(Dataset):
    """
    Memory-efficient dataset that loads AnnData files on-demand per batch
    """
    
    def __init__(self, combined_dir: str, sample_metadata_csv: str = None, 
                 sample_ids: List[str] = None, img_size: int = 256, 
                 img_channels: int = 4, transform=None, 
                 filter_min_cells: int = 3, normalize_aux: bool = False,
                 cache_metadata: bool = True, clear_cache_at_beignning: bool = False,
                 output_dir: str = None):
        """
        Args:
            cache_metadata: Whether to cache sample metadata to avoid repeated loading
        """
        self.combined_dir = combined_dir
        self.img_size = img_size
        self.img_channels = img_channels
        self.normalize_aux = normalize_aux
        self.cache_metadata = cache_metadata
        self.output_dir = output_dir
        
        if cache_metadata:
            self.metadata_cache_path = os.path.join(self.output_dir if self.output_dir is not None else self.combined_dir, 
                                                    "_dataset_metadata_cache.pkl")

        if clear_cache_at_beignning:
            if os.path.exists(self.metadata_cache_path):
                logger.info("Clearing existing metadata cache...")
                os.remove(self.metadata_cache_path)
        
        # Determine which samples to load
        if sample_metadata_csv is not None:
            metadata_df = pd.read_csv(sample_metadata_csv)
            if 'id' not in metadata_df.columns:
                raise ValueError("Metadata CSV must contain 'id' column")
            
            available_sample_ids = metadata_df['id'].tolist()
            if sample_ids is not None:
                self.sample_ids = [sid for sid in sample_ids if sid in available_sample_ids]
                missing = [sid for sid in sample_ids if sid not in available_sample_ids]
                if missing:
                    logger.warning(f"Requested samples not found in metadata: {missing}")
            else:
                self.sample_ids = available_sample_ids
        elif sample_ids is not None:
            self.sample_ids = sample_ids
        else:
            # Auto-discover samples from directory
            h5ad_files = glob.glob(os.path.join(combined_dir, "*_combined.h5ad"))
            self.sample_ids = [os.path.basename(f).replace("_combined.h5ad", "") for f in h5ad_files]
        
        logger.info(f"Initializing on-demand loading for {len(self.sample_ids)} samples")
        
        # Initialize metadata collection phase
        self._collect_dataset_metadata()
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ])
        else:
            self.transform = transform
        
        # Cache for loaded samples (LRU-style, keep most recent)
        self._sample_cache = {}
        self._cache_max_size = 2  # Keep at most 2 samples in memory
        self._cache_access_order = []
    
    def _collect_dataset_metadata(self):
        """
        First pass: collect metadata from all samples without loading full data
        This includes: patch counts, gene lists, total dataset size
        """
        if self.cache_metadata and os.path.exists(self.metadata_cache_path):
            logger.info("Loading cached dataset metadata...")
            with open(self.metadata_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.sample_metadata = cached_data['sample_metadata']
                self.patch_to_sample = cached_data['patch_to_sample']
                self.valid_patches = cached_data['valid_patches']
                self.gene_names = cached_data['gene_names']
                self.total_patches = cached_data['total_patches']
            logger.info(f"Loaded cached metadata: {self.total_patches} total patches")
            return
        
        logger.info("Collecting dataset metadata from all samples...")
        
        self.sample_metadata = {}
        self.patch_to_sample = {}
        self.valid_patches = []
        all_genes = set()
        global_patch_idx = 0
        
        for sample_id in self.sample_ids:
            adata_path = os.path.join(self.combined_dir, f"{sample_id}_combined.h5ad")
            
            if not os.path.exists(adata_path):
                logger.warning(f"Sample {sample_id} not found at {adata_path}, skipping")
                continue
            
            logger.info(f"Collecting metadata for sample {sample_id}")
            
            # Load only metadata (backed mode for memory efficiency)
            try:
                # Use backed mode to avoid loading large matrices
                adata = sc.read_h5ad(adata_path, backed='r')
                
                # Extract basic info
                patch_shape = adata.uns['patch_shape']
                n_patches = adata.n_obs
                
                # Get gene list without loading expression data
                cell_adata = adata.uns['cell_expression']  # This might still load into memory
                gene_list = cell_adata.var_names.tolist()
                all_genes.update(gene_list)
                
                # Get patch-to-cells mapping
                patch_to_cells_df = adata.uns['patch_to_cells']
                
                # Count valid patches (those with minimum cells)
                patch_cell_counts = patch_to_cells_df.groupby('patch_idx').size()
                # valid_patch_indices = patch_cell_counts[patch_cell_counts >= filter_min_cells].index.tolist()
                valid_patch_indices = patch_cell_counts.index.tolist()  # Keep all patches for now
                
                # Store metadata
                self.sample_metadata[sample_id] = {
                    'adata_path': adata_path,
                    'n_patches': n_patches,
                    'n_valid_patches': len(valid_patch_indices),
                    'valid_patch_indices': valid_patch_indices,
                    'patch_shape': patch_shape,
                    'gene_list': gene_list,
                    'n_cells': cell_adata.n_obs,
                    'n_genes': cell_adata.n_vars
                }
                
                # Create global patch mapping
                for local_idx, patch_idx in enumerate(valid_patch_indices):
                    self.patch_to_sample[global_patch_idx] = (sample_id, patch_idx)
                    self.valid_patches.append(global_patch_idx)
                    global_patch_idx += 1
                
                # Close the backed AnnData to free resources
                if hasattr(adata, 'file'):
                    adata.file.close()
                
                logger.info(f"Sample {sample_id}: {len(valid_patch_indices)} valid patches")
                
            except Exception as e:
                logger.error(f"Error collecting metadata for sample {sample_id}: {e}")
                continue
        
        # Create unified gene list
        self.gene_names = sorted(list(all_genes))
        self.total_patches = len(self.valid_patches)
        
        logger.info(f"Metadata collection complete:")
        logger.info(f"  - Total samples: {len(self.sample_metadata)}")
        logger.info(f"  - Total valid patches: {self.total_patches}")
        logger.info(f"  - Total unique genes: {len(self.gene_names)}")
        
        # Cache metadata for future runs
        if self.cache_metadata:
            cache_data = {
                'sample_metadata': self.sample_metadata,
                'patch_to_sample': self.patch_to_sample,
                'valid_patches': self.valid_patches,
                'gene_names': self.gene_names,
                'total_patches': self.total_patches
            }
            with open(self.metadata_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Cached dataset metadata for future use")
    
    def _load_sample_on_demand(self, sample_id: str):
        """Load a sample's data on-demand with LRU caching"""
        if sample_id in self._sample_cache:
            # Move to end of access order (most recently used)
            self._cache_access_order.remove(sample_id)
            self._cache_access_order.append(sample_id)
            return self._sample_cache[sample_id]
        
        # Load sample data
        logger.debug(f"Loading sample {sample_id} on-demand")
        adata_path = self.sample_metadata[sample_id]['adata_path']
        
        try:
            adata = sc.read_h5ad(adata_path)
            cell_adata = adata.uns['cell_expression']
            patch_to_cells_df = adata.uns['patch_to_cells']
            
            # Create patch-to-cells mapping
            patch_to_cells = {}
            for _, row in patch_to_cells_df.iterrows():
                patch_idx = str(row['patch_idx'])
                cell_id = str(row['cell_id'])
                if patch_idx not in patch_to_cells:
                    patch_to_cells[patch_idx] = []
                patch_to_cells[patch_idx].append(cell_id)
            
            sample_data = {
                'adata': adata,
                'cell_adata': cell_adata,
                'patch_to_cells': patch_to_cells,
                'patch_shape': self.sample_metadata[sample_id]['patch_shape']
            }
            
            # Add to cache
            self._sample_cache[sample_id] = sample_data
            self._cache_access_order.append(sample_id)
            
            # Evict oldest samples if cache is full
            while len(self._cache_access_order) > self._cache_max_size:
                oldest_sample = self._cache_access_order.pop(0)
                del self._sample_cache[oldest_sample]
                logger.debug(f"Evicted sample {oldest_sample} from cache")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error loading sample {sample_id}: {e}")
            return None
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        global_patch_idx = self.valid_patches[idx]
        sample_id, local_patch_idx = self.patch_to_sample[global_patch_idx]
        
        # Load sample data on-demand
        sample_data = self._load_sample_on_demand(sample_id)
        if sample_data is None:
            # Return dummy data if loading fails
            return self._get_dummy_item(f"{sample_id}_{local_patch_idx}")
        
        patch_idx = str(local_patch_idx)
        cell_ids = sample_data['patch_to_cells'].get(patch_idx, [])
        
        # Get patch image data
        patch_data = self._get_patch_image(sample_data['adata'], int(local_patch_idx), 
                                         sample_data['patch_shape'])
        
        # Get gene expression for all cells in this patch
        gene_exprs = []
        for cell_id in cell_ids:
            if cell_id in sample_data['cell_adata'].obs_names:
                expr = sample_data['cell_adata'][cell_id].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                else:
                    expr = expr.flatten()
                
                # Align genes with global gene list
                aligned_expr = self._align_genes_to_global(expr, 
                                                         sample_data['cell_adata'].var_names.tolist())
                gene_exprs.append(torch.tensor(aligned_expr, dtype=torch.float32))
        
        gene_exprs_tensor = torch.stack(gene_exprs) if gene_exprs else torch.zeros(1, len(self.gene_names))
        
        return {
            'patch_id': f"{sample_id}_{patch_idx}",
            'sample_id': sample_id,
            'cell_ids': cell_ids,
            'gene_expr': gene_exprs_tensor,
            'image': patch_data,
            'num_cells': len(cell_ids)
        }
    
    def _align_genes_to_global(self, expr, sample_genes):
        """Align sample-specific gene expression to global gene list"""
        aligned_expr = np.zeros(len(self.gene_names))
        sample_gene_to_idx = {gene: idx for idx, gene in enumerate(sample_genes)}
        
        for global_idx, gene in enumerate(self.gene_names):
            if gene in sample_gene_to_idx:
                sample_idx = sample_gene_to_idx[gene]
                aligned_expr[global_idx] = expr[sample_idx]
        
        return aligned_expr
    
    def _get_patch_image(self, adata, patch_idx: int, patch_shape: Dict):
        """Extract and process patch image from AnnData (same as before)"""
        try:
            if hasattr(adata.X, 'toarray'):
                patch_flat = adata.X[patch_idx].toarray().flatten()
            else:
                patch_flat = adata.X[patch_idx].flatten()
            
            patch_array = patch_flat.reshape(
                patch_shape['height'], patch_shape['width'], patch_shape['channels']
            ).astype(np.uint8)
            
            # Handle channel selection and transforms (same as before)
            return self._process_image_channels(patch_array)
                
        except Exception as e:
            logger.error(f"Error loading patch {patch_idx}: {e}")
            blank = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            return self.transform(blank) if self.transform else transforms.ToTensor()(blank)
    
    def _process_image_channels(self, patch_array):
        """Process image channels (same logic as before)"""
        if patch_array.shape[-1] > self.img_channels:
            patch_array = patch_array[:, :, :self.img_channels]
        elif patch_array.shape[-1] < self.img_channels:
            padded = np.zeros((patch_array.shape[0], patch_array.shape[1], self.img_channels), dtype=np.uint8)
            padded[:, :, :patch_array.shape[-1]] = patch_array
            patch_array = padded
        
        if self.img_channels <= 3:
            pil_img = PILImage.fromarray(patch_array.squeeze() if self.img_channels == 1 else patch_array)
            return self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        else:
            # Handle multi-channel images (same logic as before)
            rgb_img = PILImage.fromarray(patch_array[:, :, :3])
            rgb_tensor = self.transform(rgb_img) if self.transform else transforms.ToTensor()(rgb_img)
            
            aux_tensors = []
            for c in range(3, self.img_channels):
                aux_channel = patch_array[:, :, c]
                if self.normalize_aux:
                    aux_channel = self._normalize_aux_channel(aux_channel)
                aux_pil = PILImage.fromarray(aux_channel, mode='L')
                aux_tensor = self.transform(aux_pil) if self.transform else transforms.ToTensor()(aux_pil)
                aux_tensors.append(aux_tensor[0:1])
            
            return torch.cat([rgb_tensor] + aux_tensors, dim=0)
    
    def _normalize_aux_channel(self, channel):
        """Normalize auxiliary channel to 0-255 range"""
        if channel.max() > 255 or channel.dtype != np.uint8:
            channel = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-8) * 255).astype(np.uint8)
        return channel
    
    def _get_dummy_item(self, patch_id):
        """Return dummy data when sample loading fails"""
        blank_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        return {
            'patch_id': patch_id,
            'sample_id': 'dummy',
            'cell_ids': [],
            'gene_expr': torch.zeros(1, len(self.gene_names)),
            'image': self.transform(blank_img) if self.transform else transforms.ToTensor()(blank_img),
            'num_cells': 0
        }
    
    def get_sample_stats(self) -> Dict:
        """Get statistics about samples without loading them"""
        stats = {}
        for sample_id, metadata in self.sample_metadata.items():
            stats[sample_id] = {
                'n_patches': metadata['n_valid_patches'],
                'n_cells': metadata['n_cells'],
                'n_genes': metadata['n_genes']
            }
        return stats
    
    def clear_cache(self):
        """Manually clear the sample cache to free memory"""
        self._sample_cache.clear()
        self._cache_access_order.clear()
        logger.info("Cleared sample cache")


def multi_sample_hest_xenium_collate_fn(batch):
    """
    Collate function for MultiSampleHestXeniumDataset
    Handles variable number of cells per patch with proper padding
    """
    patch_ids = [item['patch_id'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    cell_ids_list = [item['cell_ids'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    num_cells = [item['num_cells'] for item in batch]
    gene_exprs = [item['gene_expr'] for item in batch]
    
    # Validate dimensions
    gene_dim = gene_exprs[0].shape[1] if len(gene_exprs) > 0 else 0
    
    # Pad gene expression tensors to max cells in batch
    max_cells = max(num_cells) if num_cells else 1
    batch_size = len(batch)
    
    padded_gene_exprs = torch.zeros(batch_size, max_cells, gene_dim, dtype=torch.float32)
    
    for i, (expr, n_cells) in enumerate(zip(gene_exprs, num_cells)):
        if n_cells > 0 and expr.shape[0] > 0:
            padded_gene_exprs[i, :min(n_cells, expr.shape[0])] = expr[:min(n_cells, max_cells)]
    
    return {
        'patch_id': patch_ids,
        'sample_id': sample_ids,  # Include sample IDs in batch
        'cell_ids': cell_ids_list,
        'gene_expr': padded_gene_exprs,
        'image': images,
        'num_cells': torch.tensor(num_cells, dtype=torch.long)
    }


class FastSeparatePatchDataset(Dataset):
    """Ultra-fast dataset using separate patch files with unified gene set"""
    
    def __init__(self, reformatted_dir, sample_metadata_csv=None, sample_ids=None, img_size=256,
                 img_channels=4, transform=None, filter_unassigned=True, 
                 min_gene_samples=1, cache_unified_genes=True):
        self.reformatted_dir = reformatted_dir
        self.img_size = img_size
        self.img_channels = img_channels
        self.filter_unassigned = filter_unassigned
        self.min_gene_samples = min_gene_samples
        self.cache_unified_genes = cache_unified_genes
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
        ])

        # Discover samples
        if sample_metadata_csv is not None:
            all_available_ids = pd.read_csv(sample_metadata_csv)['id'].tolist()
            if sample_ids is not None:
                # Filter to only requested samples that exist in metadata
                sample_ids = [sid for sid in sample_ids if sid in all_available_ids]
                missing = [sid for sid in sample_ids if sid not in all_available_ids]
                if missing:
                    logger.warning(f"Requested samples not in metadata: {missing}")
            else:
                sample_ids = all_available_ids
        elif sample_ids is None:
            sample_ids = [d for d in os.listdir(reformatted_dir)
                         if os.path.isdir(os.path.join(reformatted_dir, d))]

        # Create unified gene set FIRST
        self.unified_gene_names = self._create_unified_gene_set(sample_ids)
        logger.info(f"Created unified gene set with {len(self.unified_gene_names)} genes")

        # Load patch metadata with gene alignment
        self.patch_files = []
        self.patch_metadata = []
        self._load_patches_with_gene_alignment(sample_ids)

        logger.info(f"Loaded {len(self.patch_files)} patches from {len(sample_ids)} samples")

    def _create_unified_gene_set(self, sample_ids):
        """Create a unified gene set across all samples"""
        cache_path = os.path.join(self.reformatted_dir, "unified_genes_cache.json")
        
        # Try to load from cache
        if self.cache_unified_genes and os.path.exists(cache_path):
            logger.info("Loading unified gene set from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        logger.info("Creating unified gene set from all samples...")
        gene_counts = defaultdict(int)
        
        for sample_id in sample_ids:
            sample_dir = os.path.join(self.reformatted_dir, sample_id)
            metadata_path = os.path.join(sample_dir, "sample_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    sample_metadata = json.load(f)
                    sample_genes = sample_metadata['gene_names']
                    
                    for gene in sample_genes:
                        # Filter unassigned codewords if requested
                        if self.filter_unassigned and \
                            ('codeword' in gene.lower() or 'blank' in gene.lower()):
                            continue
                        gene_counts[gene] += 1
        
        # Keep genes that appear in at least min_gene_samples samples
        unified_genes = [gene for gene, count in gene_counts.items() 
                        if count >= self.min_gene_samples]
        unified_genes.sort()  # Ensure consistent ordering
        
        logger.info(f"Unified gene set: {len(unified_genes)} genes "
                   f"(filtered from {len(gene_counts)} total unique genes)")
        
        # Cache the result
        if self.cache_unified_genes:
            with open(cache_path, 'w') as f:
                json.dump(unified_genes, f)
            logger.info("Cached unified gene set for future use")
        
        return unified_genes

    def _load_patches_with_gene_alignment(self, sample_ids):
        """Load patch metadata and prepare gene alignment mappings"""
        self.sample_gene_mappings = {}  # sample_id -> gene alignment mapping
        
        for sample_id in sample_ids:
            sample_dir = os.path.join(self.reformatted_dir, sample_id)
            metadata_path = os.path.join(sample_dir, "patch_metadata.csv")
            sample_metadata_path = os.path.join(sample_dir, "sample_metadata.json")
            
            if os.path.exists(metadata_path) and os.path.exists(sample_metadata_path):
                # Load sample gene names
                with open(sample_metadata_path, 'r') as f:
                    sample_metadata = json.load(f)
                    sample_genes = sample_metadata['gene_names']
                
                # Create gene alignment mapping
                gene_mapping = self._create_gene_mapping(sample_genes)
                self.sample_gene_mappings[sample_id] = gene_mapping
                
                # Load patch metadata
                df = pd.read_csv(metadata_path)
                for _, row in df.iterrows():
                    self.patch_files.append(row['file_path'])
                    self.patch_metadata.append({
                        'sample_id': sample_id,
                        'patch_idx': row['patch_idx'],
                        'num_cells': row['num_cells'],
                        'gene_mapping': gene_mapping
                    })

    def _create_gene_mapping(self, sample_genes):
        """Create mapping from sample genes to unified gene indices"""
        mapping = {}
        for unified_idx, unified_gene in enumerate(self.unified_gene_names):
            if unified_gene in sample_genes:
                sample_idx = sample_genes.index(unified_gene)
                mapping[sample_idx] = unified_idx
        return mapping

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]
        metadata = self.patch_metadata[idx]
        gene_mapping = metadata['gene_mapping']

        # Load patch data
        with h5py.File(patch_file, 'r') as f:
            # Load image
            patch_image = f['image'][:]
            
            # Load cell IDs
            cell_ids = [cid.decode('utf-8') for cid in f['cell_ids'][:]]
            
            # Load gene expression and align to unified gene set
            sample_gene_expr = f['gene_expression'][:]  # Shape: [n_cells, n_sample_genes]
            
            # Create aligned gene expression matrix
            n_cells = sample_gene_expr.shape[0]
            aligned_gene_expr = np.zeros((n_cells, len(self.unified_gene_names)), dtype=np.float32)
            
            # Map sample genes to unified gene indices
            for sample_idx, unified_idx in gene_mapping.items():
                aligned_gene_expr[:, unified_idx] = sample_gene_expr[:, sample_idx]

        # Process image
        image = self._process_image(patch_image)

        # Convert aligned gene expression to tensor
        gene_expr_tensor = torch.tensor(aligned_gene_expr, dtype=torch.float32)

        return {
            'patch_id': f"{metadata['sample_id']}_{metadata['patch_idx']}",
            'sample_id': metadata['sample_id'],
            'cell_ids': cell_ids,
            'gene_expr': gene_expr_tensor,
            'image': image,
            'num_cells': len(cell_ids)
        }

    def _process_image(self, patch_array):
        """Process patch image with transforms (unchanged)"""
        if patch_array.shape[-1] > self.img_channels:
            patch_array = patch_array[:, :, :self.img_channels]
        
        if self.img_channels <= 3:
            pil_img = PILImage.fromarray(patch_array.squeeze() if self.img_channels == 1 else patch_array)
            return self.transform(pil_img)
        else:
            # Handle multi-channel as before
            rgb_img = PILImage.fromarray(patch_array[:, :, :3])
            rgb_tensor = self.transform(rgb_img)
            
            aux_tensors = []
            for c in range(3, self.img_channels):
                aux_channel = patch_array[:, :, c]
                aux_pil = PILImage.fromarray(aux_channel, mode='L')
                aux_tensor = self.transform(aux_pil)
                aux_tensors.append(aux_tensor[0:1])
            
            return torch.cat([rgb_tensor] + aux_tensors, dim=0)

    @property
    def gene_names(self):
        """Return unified gene names for model initialization"""
        return self.unified_gene_names


def fast_separate_patch_collate_fn(batch):
    """
    Collate function for FastSeparatePatchDataset with unified gene dimensions
    """
    patch_ids = [item['patch_id'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    cell_ids_list = [item['cell_ids'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    num_cells = [item['num_cells'] for item in batch]
    gene_exprs = [item['gene_expr'] for item in batch]

    # All gene expressions should now have the same gene dimension
    gene_dim = gene_exprs[0].shape[1] if len(gene_exprs) > 0 else 0
    
    # Verify all samples have consistent gene dimensions
    for i, expr in enumerate(gene_exprs):
        if expr.shape[1] != gene_dim:
            raise ValueError(f"Inconsistent gene dimensions in batch: "
                           f"expected {gene_dim}, got {expr.shape[1]} for sample {i}")

    # Handle padding for variable number of cells per patch
    max_cells = max(num_cells) if num_cells else 1
    batch_size = len(batch)
    
    padded_gene_exprs = torch.zeros(batch_size, max_cells, gene_dim, dtype=torch.float32)
    
    for i, (expr, n_cells) in enumerate(zip(gene_exprs, num_cells)):
        if n_cells > 0 and expr.shape[0] > 0:
            cells_to_copy = min(n_cells, expr.shape[0], max_cells)
            padded_gene_exprs[i, :cells_to_copy] = expr[:cells_to_copy]

    return {
        'patch_id': patch_ids,
        'sample_id': sample_ids,
        'cell_ids': cell_ids_list,
        'gene_expr': padded_gene_exprs,
        'image': images,
        'num_cells': torch.tensor(num_cells, dtype=torch.long)
    }
