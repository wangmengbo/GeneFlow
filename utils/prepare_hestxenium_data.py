import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Tuple, Union, Optional
import h5py
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from hest import iter_hest
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image as PILImage
from shapely.geometry import box, Polygon, Point
import shapely.geometry
import shapely.affinity
import anndata as ad
import random
import cv2
from rtree import index # Install with: pip install rtree
from scipy.sparse import csr_matrix
from openslide import OpenSlide
import traceback
from matplotlib.path import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def polygon_to_mask(polygon, patch_origin, patch_size):
    """Convert a single polygon to binary mask relative to patch coordinates"""
    # Translate polygon to patch-relative coordinates
    polygon_shifted = shapely.affinity.translate(
        polygon, xoff=-patch_origin[0], yoff=-patch_origin[1]
    )
    coords = np.array(polygon_shifted.exterior.coords)
    
    # Create grid of pixel coordinates
    x, y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    
    # Use matplotlib Path to check which pixels are inside polygon
    path = Path(coords)
    mask = path.contains_points(points).reshape(patch_size, patch_size).astype(np.uint8)
    return mask


def create_patch_mask(patch_bounds, cell_polygons, patch_size):
    """Create combined binary mask for all polygons intersecting the patch"""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    patch_origin = (patch_bounds.bounds[0], patch_bounds.bounds[1]) 
    
    for cell_id, polygon in cell_polygons.items():
        if not patch_bounds.intersects(polygon):
            continue
        poly_mask = polygon_to_mask(polygon, patch_origin, patch_size)
        mask = np.maximum(mask, poly_mask)  # Union of all polygons
    
    return mask
    

def load_wsi_directly(wsi_path):
    """Load WSI directly from file path"""
    wsi = OpenSlide(wsi_path)
    return wsi


def create_spatial_index(cell_polygons):
    """Create R-tree spatial index for ultra-fast spatial queries"""
    tmp = {k:cell_polygons[k] for i,k in enumerate(cell_polygons.keys()) if i<5}
    logger.debug(f"cell_polygons[:5]={tmp}")
    spatial_idx = index.Index()
    cell_id_to_idx = {}
    for i, (cell_id, polygon) in enumerate(cell_polygons.items()):
        bounds = polygon.bounds # (minx, miny, maxx, maxy)
        spatial_idx.insert(i, bounds)
        cell_id_to_idx[i] = cell_id
    logger.info(f"Created spatial index with {len(cell_polygons)} polygons")
    return spatial_idx, cell_id_to_idx


def fast_filter_cells_spatial_index(spatial_idx, cell_id_to_idx, cell_polygons, patch_bounds,
                                    completely_within=False):
    """Ultra-fast cell filtering using spatial index + precise polygon check"""
    candidate_indices = list(spatial_idx.intersection(patch_bounds.bounds))
    logger.debug(f"Found {len(candidate_indices)} candidate cells in patch bounds {patch_bounds}")
    logger.debug(f"spatial_idx={spatial_idx}")
    
    cells_in_patch = []
    for idx in candidate_indices:
        cell_id = cell_id_to_idx[idx]
        cell_polygon = cell_polygons[cell_id]
        
        if completely_within and patch_bounds.contains(cell_polygon):
            cells_in_patch.append(cell_id)
        elif patch_bounds.intersects(cell_polygon):
            cells_in_patch.append(cell_id)
    
    return cells_in_patch


def process_single_cell_batch(wsi_path, sample_id, cells_batch, patch_size,
                             cell_centroids, cell_expr_df, gene_names,
                             output_dir, compression='lzf'):
    """Process a batch of single cells and save directly to separate files"""
    results = []
    
    try:
        # Load WSI directly
        wsi = load_wsi_directly(wsi_path)
        
        for cell_id in cells_batch:
            try:
                # Get cell centroid
                centroid_x, centroid_y = cell_centroids[cell_id]
                
                # Calculate patch coordinates (centered on cell)
                half_patch = patch_size // 2
                x_start = int(centroid_x - half_patch)
                y_start = int(centroid_y - half_patch)
                
                # Ensure patch is within WSI bounds
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(wsi.dimensions[0], x_start + patch_size)
                y_end = min(wsi.dimensions[1], y_start + patch_size)
                
                # Adjust start coordinates if patch would exceed WSI bounds
                if x_end - x_start < patch_size:
                    x_start = max(0, x_end - patch_size)
                if y_end - y_start < patch_size:
                    y_start = max(0, y_end - patch_size)
                
                # Extract patch from WSI
                patch_region = wsi.read_region((x_start, y_start), 0, (patch_size, patch_size))
                
                # Handle different return types
                if isinstance(patch_region, np.ndarray):
                    patch_array = patch_region
                    if patch_array.shape[-1] == 4:  # RGBA
                        patch_array = patch_array[..., :3]
                else:
                    patch_array = np.array(patch_region.convert('RGB'))
                
                # Ensure correct size
                if patch_array.shape[:2] != (patch_size, patch_size):
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    h, w = patch_array.shape[:2]
                    padded_patch[:min(h, patch_size), :min(w, patch_size)] = \
                        patch_array[:min(h, patch_size), :min(w, patch_size)]
                    patch_array = padded_patch
                
                patch_array = patch_array.astype(np.uint8)
                
                # Extract gene expression for this cell
                cell_expr = cell_expr_df.loc[[cell_id]] if cell_id in cell_expr_df.index else pd.DataFrame()
                
                # Save patch directly to separate file format using cell_id as filename
                patch_file = os.path.join(output_dir, "patches", f"cell_{cell_id}.h5")
                
                with h5py.File(patch_file, 'w') as f:
                    # Save image
                    f.create_dataset('image', data=patch_array, compression=compression)
                    
                    # Save cell ID
                    f.create_dataset('cell_id', data=cell_id.encode('utf-8'))
                    
                    # Save gene expression
                    if len(cell_expr) > 0:
                        f.create_dataset('gene_expression',
                                       data=cell_expr.values.astype(np.float32),
                                       compression=compression)
                    else:
                        f.create_dataset('gene_expression',
                                       data=np.array([]).reshape(0, len(gene_names)),
                                       compression=compression)
                    
                    # Save metadata
                    f.attrs['cell_id'] = cell_id
                    f.attrs['sample_id'] = sample_id
                    f.attrs['centroid_x'] = centroid_x
                    f.attrs['centroid_y'] = centroid_y
                    f.attrs['patch_coordinates'] = f"{x_start},{y_start}"
                
                results.append({
                    'cell_id': cell_id,
                    'file_path': patch_file,
                    'centroid': (centroid_x, centroid_y),
                    'patch_coordinates': (x_start, y_start)
                })
                
            except Exception as e:
                logger.debug(f"Failed to process cell {cell_id}: {e}")
                continue
        
        # Close WSI if it has a close method
        if hasattr(wsi, 'close'):
            wsi.close()
            
    except Exception as e:
        logger.error(f"Failed to process batch for WSI {wsi_path}: {e}")
        return []
    
    return results


def process_patch_batch_direct_save(wsi_path, sample_id, patch_coords_batch, patch_size,
                                    spatial_idx_data, cell_polygons_data, cell_expr_df, gene_names,
                                    output_dir, min_cells_per_patch=5, completely_within=False,
                                    compression='lzf'):
    """Process a batch of patches and save directly to separate files"""
    logger.debug(f"process_patch_batch_direct_save using min_cells_per_patch={min_cells_per_patch}")

    # Rebuild spatial index in worker process
    spatial_idx = index.Index()
    cell_id_to_idx = {}
    for i, (cell_id, polygon) in enumerate(cell_polygons_data.items()):
        bounds = polygon.bounds
        spatial_idx.insert(i, bounds)
        cell_id_to_idx[i] = cell_id
    
    cell_polygons = cell_polygons_data
    results = []
    
    try:
        # Load WSI directly
        wsi = load_wsi_directly(wsi_path)
        
        for patch_idx, (x_start, y_start, global_idx) in enumerate(patch_coords_batch):
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Create patch boundary
            patch_bounds = box(x_start, y_start, x_end, y_end)
            logger.debug(f"Processing patch at ({x_start}, {y_start}) with bounds {patch_bounds.bounds}")
            
            # Fast cell filtering using spatial index
            cells_in_patch = fast_filter_cells_spatial_index(
                spatial_idx, cell_id_to_idx, cell_polygons, patch_bounds,
                completely_within=completely_within
            )
            
            if len(cells_in_patch) < min_cells_per_patch:
                logger.info(f"Skipping patch at ({x_start}, {y_start}) with {len(cells_in_patch)} cells")
                continue
            
            try:
                # Extract patch from WSI
                patch_region = wsi.read_region((x_start, y_start), 0, (patch_size, patch_size))
                
                # Handle different return types
                if isinstance(patch_region, np.ndarray):
                    patch_array = patch_region
                    if patch_array.shape[-1] == 4:  # RGBA
                        patch_array = patch_array[..., :3]
                else:
                    patch_array = np.array(patch_region.convert('RGB'))
                
                # Ensure correct size
                if patch_array.shape[:2] != (patch_size, patch_size):
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    h, w = patch_array.shape[:2]
                    padded_patch[:min(h, patch_size), :min(w, patch_size)] = \
                        patch_array[:min(h, patch_size), :min(w, patch_size)]
                    patch_array = padded_patch
                
                patch_array = patch_array.astype(np.uint8)
                
                # Add polygon mask generation:
                patch_bounds = box(x_start, y_start, x_end, y_end)
                mask = create_patch_mask(patch_bounds, cell_polygons_data, patch_size)

                # Convert mask to 0-255 scale and add as 4th channel
                mask_channel = (mask * 255).astype(np.uint8)
                patch_array = np.concatenate([patch_array, mask_channel[..., None]], axis=2)

                # Now patch_array has shape (patch_size, patch_size, 4) instead of (patch_size, patch_size, 3)

                # Extract gene expression for cells in this patch
                cells_expr = cell_expr_df.loc[cells_in_patch] if cells_in_patch else pd.DataFrame()
                
                # Save patch directly to separate file format
                # global_patch_idx = patch_coords_batch[0][2] + patch_idx  # Assuming we pass global index
                global_patch_idx = global_idx
                patch_file = os.path.join(output_dir, "patches", f"patch_{global_patch_idx:06d}.h5")
                
                with h5py.File(patch_file, 'w') as f:
                    # Save image
                    f.create_dataset('image', data=patch_array, compression=compression)
                    
                    # Save cell IDs
                    f.create_dataset('cell_ids',
                                   data=[cell_id.encode('utf-8') for cell_id in cells_in_patch])
                    
                    # Save gene expression
                    if len(cells_expr) > 0:
                        f.create_dataset('gene_expression',
                                       data=cells_expr.values.astype(np.float32),
                                       compression=compression)
                    else:
                        f.create_dataset('gene_expression',
                                       data=np.array([]).reshape(0, len(gene_names)),
                                       compression=compression)
                    
                    # Save metadata
                    f.attrs['patch_idx'] = global_patch_idx
                    f.attrs['num_cells'] = len(cells_in_patch)
                    f.attrs['sample_id'] = sample_id
                    f.attrs['coordinates'] = f"{x_start},{y_start}"
                
                results.append({
                    'patch_idx': global_patch_idx,
                    'file_path': patch_file,
                    'num_cells': len(cells_in_patch),
                    'cells': cells_in_patch,
                    'coordinates': (x_start, y_start)
                })
                
            except Exception as e:
                logger.debug(f"Failed to process patch at ({x_start}, {y_start}): {e}")
                continue
        
        # Close WSI if it has a close method
        if hasattr(wsi, 'close'):
            wsi.close()
            
    except Exception as e:
        logger.error(f"Failed to process batch for WSI {wsi_path}: {e}")
        return []
    
    return results


def extract_single_cell_patches(sample_id, st, adata, cell_centroids, output_dir,
                               patch_size=256, max_workers=None, batch_size=100,
                               test=False, n_tests=10, wsi_base_path=None,
                               compression='lzf'):
    """Extract patches centered on individual qualified cells"""
    
    # Create output directories
    sample_output_dir = os.path.join(output_dir, sample_id)
    patches_dir = os.path.join(sample_output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    
    # Optimal worker calculation
    if max_workers is None:
        cpu_cores = os.cpu_count() or 4
        max_workers = min(cpu_cores * 3, 48)
    logger.info(f"Using {max_workers} workers with batch size {batch_size}")
    
    # Get qualified cells (those that passed gene expression filters)
    qualified_cells = list(adata.obs_names)
    logger.info(f"Processing {len(qualified_cells)} qualified cells")
    
    # Test mode: use subset
    if test:
        qualified_cells = random.sample(qualified_cells, min(n_tests, len(qualified_cells)))
        logger.info(f"Test mode: processing {len(qualified_cells)} random cells")
    
    # Get gene expression data
    cell_expr_df = adata.to_df()
    gene_names = adata.var_names.tolist()
    
    # Create batches for processing
    batches = [qualified_cells[i:i + batch_size] for i in range(0, len(qualified_cells), batch_size)]
    logger.info(f"Created {len(batches)} batches")
    
    # Construct WSI path
    if wsi_base_path is None:
        wsi_base_path = "/home/wang4887/HnE_RNA/GeneFlow/data/HEST-1k/data/wsis"
    wsi_path = os.path.join(wsi_base_path, f"{sample_id}.tif")
    
    # Verify WSI file exists
    assert os.path.exists(wsi_path), f"WSI file not found: {wsi_path}"
    logger.info(f"Using WSI path: {wsi_path}")
    
    valid_patches = []
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        futures = [
            executor.submit(
                process_single_cell_batch,
                wsi_path,
                sample_id,
                batch,
                patch_size,
                cell_centroids,
                cell_expr_df,
                gene_names,
                sample_output_dir,
                compression
            )
            for batch in batches
        ]
        
        # Collect results with progress bar
        with tqdm(total=len(futures), desc='Processing single-cell patches') as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    valid_patches.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                pbar.update(1)
    
    if not valid_patches:
        raise ValueError(f"No valid patches extracted for qualified cells")
    
    # Sort patches by cell_id for consistent ordering
    valid_patches.sort(key=lambda x: x['cell_id'])
    
    # Save sample metadata
    sample_metadata = convert_numpy_types({
        'sample_id': sample_id,
        'mode': 'single',
        'n_patches': len(valid_patches),
        'patch_shape': {
            'height': patch_size,
            'width': patch_size,
            'channels': 3  # RGB channels
        },
        'gene_names': gene_names,
        'n_genes': len(gene_names),
        'total_qualified_cells': len(qualified_cells)
    })
    
    # Save metadata files
    with open(os.path.join(sample_output_dir, "sample_metadata.json"), 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    # Create patch metadata DataFrame
    patch_metadata_df = pd.DataFrame(valid_patches)
    patch_metadata_df.to_csv(os.path.join(sample_output_dir, "patch_metadata.csv"), index=False)
    
    logger.info(f"Sample {sample_id}: {len(valid_patches)} single-cell patches saved")
    return len(valid_patches)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def extract_xenium_patches_direct_save(sample_id, st, adata, cell_centroids, cell_polygons,
                                       wsi_width, wsi_height, output_dir,
                                       patch_size=256, stride=128,
                                       min_cells_per_patch=5, completely_within=False,
                                       max_workers=None, batch_size=100,
                                       test=False, n_tests=10, wsi_base_path=None,
                                       compression='lzf'):
    """Extract patches and save directly to separate patch format"""
    logger.info(f"extract_xenium_patches_direct_save received min_cells_per_patch={min_cells_per_patch}")

    # Create output directories
    sample_output_dir = os.path.join(output_dir, sample_id)
    patches_dir = os.path.join(sample_output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    
    # Optimal worker calculation
    if max_workers is None:
        cpu_cores = os.cpu_count() or 4
        max_workers = min(cpu_cores * 3, 48)
    logger.info(f"Using {max_workers} workers with batch size {batch_size}")
    
    # Create spatial index
    logger.info("Creating spatial index...")
    spatial_idx, cell_id_to_idx = create_spatial_index(cell_polygons)
    spatial_idx_data = (spatial_idx, cell_id_to_idx)
    
    # Generate all patch coordinates with global indices
    patch_coords = []
    global_patch_idx = 0
    for y in range(0, wsi_height - patch_size + 1, stride):
        for x in range(0, wsi_width - patch_size + 1, stride):
            patch_coords.append((x, y, global_patch_idx))
            global_patch_idx += 1
    
    logger.info(f"Created {len(patch_coords)} patch coordinates")
    
    # Test mode: use subset
    if test:
        patch_coords = random.sample(patch_coords, min(n_tests, len(patch_coords)))
        logger.info(f"Test mode: processing {len(patch_coords)} random patches")
    
    # Get gene expression data
    cell_expr_df = adata.to_df()
    gene_names = adata.var_names.tolist()
    
    # Create batches for processing
    batches = [patch_coords[i:i + batch_size] for i in range(0, len(patch_coords), batch_size)]
    logger.info(f"Created {len(batches)} batches")
    
    # Construct WSI path
    if wsi_base_path is None:
        wsi_base_path = "/home/wang4887/HnE_RNA/GeneFlow/data/HEST-1k/data/wsis"
    wsi_path = os.path.join(wsi_base_path, f"{sample_id}.tif")
    
    # Verify WSI file exists
    assert os.path.exists(wsi_path), f"WSI file not found: {wsi_path}"
    logger.info(f"Using WSI path: {wsi_path}")
    
    valid_patches = []
    
    # Process batches in parallel - MODIFIED to save directly
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs with direct saving
        futures = [
            executor.submit(
                process_patch_batch_direct_save,
                wsi_path,
                sample_id,
                batch,
                patch_size,
                spatial_idx_data,
                cell_polygons,
                cell_expr_df,
                gene_names,
                sample_output_dir,
                min_cells_per_patch,
                completely_within,
                compression
            )
            for batch in batches
        ]
        
        # Collect results with progress bar
        with tqdm(total=len(batches), desc='Processing and saving patch batches') as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    if batch_results:
                        valid_patches.extend(batch_results)
                    else:
                        logger.warning("Received empty batch_results from worker")
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                pbar.update(1)
    
    if not valid_patches:
        raise ValueError(f"No valid patches extracted. Try reducing min_cells_per_patch "
                        f"(current: {min_cells_per_patch}) or increasing patch_size "
                        f"(current: {patch_size})")
    
    # Sort patches by coordinates for consistent ordering
    valid_patches.sort(key=lambda x: x['coordinates'])
    
    # Save sample metadata
    sample_metadata = convert_numpy_types({
        'sample_id': sample_id,
        'mode': 'multi',
        'n_patches': len(valid_patches),
        'patch_shape': {
            'height': patch_size,
            'width': patch_size,
            'channels': 4  # RGB + mask channel
        },
        'gene_names': gene_names,
        'n_genes': len(gene_names),
        'total_cells': len(cell_expr_df)
    })
    
    # Save metadata files
    with open(os.path.join(sample_output_dir, "sample_metadata.json"), 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    # Create patch metadata DataFrame
    patch_metadata_df = pd.DataFrame(valid_patches)
    patch_metadata_df.to_csv(os.path.join(sample_output_dir, "patch_metadata.csv"), index=False)
    
    logger.info(f"Sample {sample_id}: {len(valid_patches)} patches saved directly to separate format")
    return valid_patches


def create_single_cell_expression_matrix(transcript_df):
    """Convert raw transcript data to single-cell gene expression matrix"""
    for i in ['cell_id', 'feature_name']:
        transcript_df[i] = transcript_df[i].astype(str)
        if transcript_df[i].str.startswith('b\'').any():
            transcript_df[i] = transcript_df[i].str.decode('utf-8')
    
    expression_counts = transcript_df.groupby(['cell_id', 'feature_name']).size().reset_index(name='counts')
    cell_gene_matrix = expression_counts.pivot_table(
        index='cell_id',
        columns='feature_name',
        values='counts',
        fill_value=0
    )
    
    return cell_gene_matrix


def add_coordinates_to_metadata_within_prepare(data_dir, sample_id, output_dir, patch_size, stride, wsi_width, wsi_height):
    """
    Add patch coordinates to patch_metadata.csv after extraction (sliding window mode).
    """
    sample_output_dir = os.path.join(output_dir, sample_id)
    metadata_file = os.path.join(sample_output_dir, "patch_metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"patch_metadata.csv not found at {metadata_file}")

    patch_metadata_df = pd.read_csv(metadata_file)

    patch_coords = []
    global_patch_idx = 0
    for y in range(0, wsi_height - patch_size + 1, stride):
        for x in range(0, wsi_width - patch_size + 1, stride):
            patch_coords.append((x, y))
            global_patch_idx += 1

    coord_dict = {i: coord for i, coord in enumerate(patch_coords)}
    patch_metadata_df['coordinates'] = patch_metadata_df['patch_idx'].map(coord_dict)
    patch_metadata_df['x_coord'] = patch_metadata_df['coordinates'].apply(lambda x: x[0] if x else None)
    patch_metadata_df['y_coord'] = patch_metadata_df['coordinates'].apply(lambda x: x[1] if x else None)
    patch_metadata_df.to_csv(metadata_file, index=False)
    logging.info(f"Added patch coordinates to {metadata_file}")


def prepare_xenium_dataset_direct_save(data_dir, sample_id, output_dir, mode='multi',
                                       max_workers=None, min_genes=20, min_cells=3,
                                       batch_size=100, wsi_base_path=None,
                                       compression='lzf', **kwargs):
    """Optimized pipeline that saves directly to separate patch format"""
    
    for st in iter_hest(data_dir, id_list=[sample_id], load_transcripts=True):
        # Preprocess gene expression data
        single_cell_expr = create_single_cell_expression_matrix(st.transcript_df)
        logger.info(f"single_cell_expr.head={single_cell_expr.head()}")
        
        cell_adata = ad.AnnData(
            X=single_cell_expr.values,
            obs=pd.DataFrame(index=single_cell_expr.index),
            var=pd.DataFrame(index=single_cell_expr.columns)
        )
        
        n_cells_orig = cell_adata.n_obs
        logger.info(f"Created AnnData with {n_cells_orig} cells and {cell_adata.n_vars} genes")
        
        sc.pp.calculate_qc_metrics(cell_adata, percent_top=None, log1p=False, inplace=True)
        genes_per_cell = cell_adata.obs['n_genes_by_counts']
        p5 = np.percentile(genes_per_cell, 5)
        p95 = np.percentile(genes_per_cell, 95)
        logger.info(f"5th percentile: {p5:.0f} genes")
        logger.info(f"95th percentile: {p95:.0f} genes")
        # Create boolean mask for cells within 5th-95th percentile
        # Filter the data
        cell_adata = cell_adata[(cell_adata.obs['n_genes_by_counts'] >= p5) & 
                                (cell_adata.obs['n_genes_by_counts'] <= p95), :]
        
        logger.info(f"Filtered cells: {cell_adata.n_obs}")
        logger.info(f"Cells removed: {n_cells_orig - cell_adata.n_obs} ({((n_cells_orig - cell_adata.n_obs)/n_cells_orig)*100:.1f}%)")

        # Apply gene expression filters
        sc.pp.filter_cells(cell_adata, min_genes=min_genes)
        logger.info(f"Filtered cells to {cell_adata.n_obs} with at least {min_genes} genes")
        
        sc.pp.normalize_total(cell_adata, target_sum=1e6)
        sc.pp.log1p(cell_adata)
        
        # Get cell shapes for spatial operations
        cell_shapes_df = st.get_shapes('xenium_cell', 'he').shapes
        logger.info(f"cell_shapes_df.head={cell_shapes_df.head()}")
        
        # Handle byte string indices if needed
        cell_shapes_df.index = cell_shapes_df.index.astype(str)
        if cell_shapes_df.index.str.startswith('b\'').any():
            cell_shapes_df.index = cell_shapes_df.index.str.decode('utf-8')
            logger.info("Converted cell shape indices from bytes to utf-8 strings")
        
        # Filter cell_shapes_df with cells included in cell_adata
        cell_shapes_df = cell_shapes_df[cell_shapes_df.index.isin(cell_adata.obs_names)]
        logger.info(f"Loaded {len(cell_shapes_df)} cell polygons")
        
        # Calculate centroids and polygons
        cell_centroids = {}
        cell_polygons = {}
        
        for cell_idx, (idx, row) in enumerate(cell_shapes_df.iterrows()):
            polygon = row['geometry']
            centroid = polygon.centroid
            cell_id_str = str(idx)
            
            cell_centroids[cell_id_str] = (centroid.x, centroid.y)
            cell_polygons[cell_id_str] = polygon
        
        # Choose processing mode
        if mode == 'single':
            # Single cell mode: extract patches centered on individual cells
            # Filter kwargs for single mode (remove multi-mode specific parameters)
            single_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['stride', 'min_cells_per_patch', 'completely_within']}
            
            n_patches = extract_single_cell_patches(
                sample_id, st, cell_adata, cell_centroids, output_dir,
                max_workers=max_workers,
                batch_size=batch_size,
                wsi_base_path=wsi_base_path,
                compression=compression,
                **single_kwargs
            )
            logger.info(f"Successfully saved {n_patches} single-cell patches")
            
        else:
            # Multi cell mode: sliding window approach (original implementation)
            # Get WSI dimensions
            wsi_width = st.wsi.width
            wsi_height = st.wsi.height
            
            logger.info(f"Calling extract_xenium_patches_direct_save with min_cells_per_patch={kwargs.get('min_cells_per_patch', 'NOT SET')}")

            # Extract patches and save directly to separate format
            valid_patches = extract_xenium_patches_direct_save(
                sample_id, st, cell_adata, cell_centroids, cell_polygons,
                wsi_width, wsi_height, output_dir,
                max_workers=max_workers,
                batch_size=batch_size,
                wsi_base_path=wsi_base_path,
                compression=compression,
                **kwargs
            )
            logger.info(f"Successfully saved {len(valid_patches)} multi-cell patches")

            # Add coordinates to metadata
            add_coordinates_to_metadata_within_prepare(
                data_dir, sample_id, output_dir, 
                kwargs.get('patch_size', None), 
                kwargs.get('stride', None), 
                wsi_width, wsi_height
            )
        
        return len(valid_patches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Xenium dataset preparation to separate patch format")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--sample_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--mode', type=str, default='multi', choices=['single', 'multi'], 
                       help='Processing mode: single (cell-centered) or multi (sliding window)')
    parser.add_argument('--min_cells_per_patch', type=int, default=3)
    parser.add_argument('--completely_within', action='store_true',
                       help='If true, only include cells completely within the patch')
    parser.add_argument('--max_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_tests', type=int, default=10, help='Number of patches to test in debug mode')
    parser.add_argument('--wsi_base_path', type=str,
                       default="/depot/natallah/data/Mengbo/HnE_RNA/data/HEST-1k/data/wsis",
                       help='Base path for WSI files')
    parser.add_argument('--compression', type=str, default='lzf', help='HDF5 compression type')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose logging')
    
    args = parser.parse_args()
    logger.info(f"args={args}")
    
    if args.stride is None:
        args.stride = args.patch_size // 2
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Run direct preparation to separate format
    n_patches = prepare_xenium_dataset_direct_save(
        data_dir=args.data_dir,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        mode=args.mode,
        min_cells_per_patch=args.min_cells_per_patch,
        completely_within=args.completely_within,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        test=args.test,
        n_tests=args.n_tests,
        wsi_base_path=args.wsi_base_path,
        compression=args.compression,
    )
    
    logger.info(f"Direct preparation complete! Saved {n_patches} patches in {args.mode} mode.")
