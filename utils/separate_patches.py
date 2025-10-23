import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import json
import argparse
import logging
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    

def reformat_to_separate_patches(combined_dir, output_dir, sample_id=None, compression='lzf'):
    """
    Reformat combined AnnData files to separate patch storage
    
    Args:
        combined_dir: Directory containing *_combined.h5ad files
        output_dir: Directory to save reformatted data
        compression: HDF5 compression ('lzf', 'gzip', or None)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all combined files
    if sample_id is None:
        combined_files = glob.glob(os.path.join(combined_dir, "*_combined.h5ad"))
    else:
        combined_files = [os.path.join(combined_dir, f"{sample_id}_combined.h5ad")]
    logger.info(f"Found {len(combined_files)} combined files in {combined_dir}")

    for adata_path in tqdm(combined_files, desc="Reformatting samples"):
        sample_id = os.path.basename(adata_path).replace("_combined.h5ad", "")
        sample_output_dir = os.path.join(output_dir, sample_id)
        patches_dir = os.path.join(sample_output_dir, "patches")
        os.makedirs(patches_dir, exist_ok=True)
        
        logger.info(f"Processing sample {sample_id}...")
        
        # Load original data
        adata = sc.read_h5ad(adata_path)
        cell_adata = adata.uns['cell_expression']
        patch_to_cells_df = adata.uns['patch_to_cells']
        patch_shape = adata.uns['patch_shape']
        
        # Get patches data
        if hasattr(adata.X, 'toarray'):
            patches_flat = adata.X.toarray()
        else:
            patches_flat = adata.X
            
        # Reshape patches
        n_patches = patches_flat.shape[0]
        patches_array = patches_flat.reshape(
            n_patches, 
            patch_shape['height'], 
            patch_shape['width'], 
            patch_shape['channels']
        )
        
        # Create patch-to-cells mapping
        patch_to_cells = {}
        for _, row in patch_to_cells_df.iterrows():
            patch_idx = str(row['patch_idx'])
            cell_id = str(row['cell_id'])
            if patch_idx not in patch_to_cells:
                patch_to_cells[patch_idx] = []
            patch_to_cells[patch_idx].append(cell_id)
        
        # Get cell expression data
        cell_expr_df = cell_adata.to_df()
        gene_names = cell_adata.var_names.tolist()
        
        # Save each patch separately
        patch_metadata = []
        
        for patch_idx in range(n_patches):
            patch_idx_str = str(patch_idx)
            
            # Get cells for this patch
            cells_in_patch = patch_to_cells.get(patch_idx_str, [])
            
            if len(cells_in_patch) == 0:
                continue
                
            # Extract patch image
            patch_image = patches_array[patch_idx]
            
            # Extract gene expression for cells in this patch
            cells_expr = cell_expr_df.loc[cells_in_patch] if cells_in_patch else pd.DataFrame()
            
            # Save patch data
            patch_file = os.path.join(patches_dir, f"patch_{patch_idx:06d}.h5")
            
            with h5py.File(patch_file, 'w') as f:
                # Save image
                f.create_dataset('image', data=patch_image, 
                               compression=compression)
                
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
                f.attrs['patch_idx'] = patch_idx
                f.attrs['num_cells'] = len(cells_in_patch)
                f.attrs['sample_id'] = sample_id
            
            patch_metadata.append({
                'patch_idx': patch_idx,
                'file_path': patch_file,
                'num_cells': len(cells_in_patch),
                'cells': cells_in_patch
            })
        
        sample_metadata = convert_numpy_types({
            'sample_id': sample_id,
            'n_patches': len(patch_metadata),
            'patch_shape': patch_shape,  # Now this will be automatically converted
            'gene_names': gene_names,
            'n_genes': len(gene_names),
            'total_cells': len(cell_expr_df)
        })
        
        # Save metadata files
        with open(os.path.join(sample_output_dir, "sample_metadata.json"), 'w') as f:
            json.dump(sample_metadata, f, indent=2)
            
        patch_metadata_df = pd.DataFrame(patch_metadata)
        patch_metadata_df.to_csv(os.path.join(sample_output_dir, "patch_metadata.csv"), index=False)
        
        logger.info(f"Sample {sample_id}: {len(patch_metadata)} patches saved")
    
    logger.info(f"Reformatting complete! Data saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reformat combined AnnData files to separate patches")
    parser.add_argument('--combined_dir', type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/processed_data/hest1k_xenium", help="Directory containing *_combined.h5ad files")
    parser.add_argument('--output_dir', type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/processed_data/hest1k_xenium", help="Directory to save reformatted data")
    parser.add_argument('--sample_id', type=str, default=None, help="Specific sample ID to reformat (default: None, reformat all)")
    parser.add_argument('--compression', type=str, default='lzf', help="HDF5 compression type (default: lzf)")

    args = parser.parse_args()

    # Reformat your existing data
    reformat_to_separate_patches(
        combined_dir=args.combined_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        compression=args.compression
    )
