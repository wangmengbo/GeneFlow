#!/usr/bin/env python3

import pandas as pd
import os
import logging
import argparse
from hest import iter_hest
import sys
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_coordinates_to_existing_metadata(data_dir, sample_id, output_dir, 
                                       patch_size=256, stride=128):
    """Add coordinates to existing patch_metadata.csv"""
    try:
        # Get WSI dimensions
        logger.info(f"Loading WSI dimensions for {sample_id}")
        for st in iter_hest(data_dir, id_list=[sample_id], load_transcripts=False):
            wsi_width = st.wsi.width
            wsi_height = st.wsi.height
            break
        
        logger.info(f"WSI dimensions: {wsi_width} x {wsi_height}")
        
        # Load existing metadata
        sample_output_dir = os.path.join(output_dir, sample_id)
        metadata_file = os.path.join(sample_output_dir, "patch_metadata.csv")
        
        if not os.path.exists(metadata_file):
            e = f"Metadata file not found: {metadata_file}"
            logger.error(e)
            return FileNotFoundError(e)
        
        patch_metadata_df = pd.read_csv(metadata_file)
        logger.info(f"Loaded {len(patch_metadata_df)} patches from metadata")
        
        # Regenerate patch coordinates using same logic as original extraction
        patch_coords = []
        global_patch_idx = 0
        for y in range(0, wsi_height - patch_size + 1, stride):
            for x in range(0, wsi_width - patch_size + 1, stride):
                patch_coords.append((x, y))
                global_patch_idx += 1
        
        logger.info(f"Generated {len(patch_coords)} coordinate pairs")
        
        # Create mapping from patch_idx to coordinates
        coord_dict = {i: coord for i, coord in enumerate(patch_coords)}
        
        # Add coordinates columns to dataframe
        patch_metadata_df['coordinates'] = patch_metadata_df['patch_idx'].map(coord_dict)
        patch_metadata_df['x_coord'] = patch_metadata_df['coordinates'].apply(lambda x: x[0] if x else None)
        patch_metadata_df['y_coord'] = patch_metadata_df['coordinates'].apply(lambda x: x[1] if x else None)
        
        # Backup original file
        backup_file = metadata_file + ".backup"
        if os.path.exists(metadata_file):
            os.rename(metadata_file, backup_file)
            logger.info(f"Backed up original metadata to {backup_file}")
        
        # Save updated metadata
        patch_metadata_df.to_csv(metadata_file, index=False)
        logger.info(f"Updated metadata saved with coordinates for {len(patch_metadata_df)} patches")
        
        # Show sample of updated data
        print("\nSample of updated metadata:")
        print(patch_metadata_df[['patch_idx', 'coordinates', 'x_coord', 'y_coord', 'num_cells']].head())
        
        return patch_metadata_df
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add coordinates to existing patch metadata")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='HEST data directory')
    parser.add_argument('--sample_id', type=str, default=None, nargs='*',
                       help='Sample ID (e.g., TENX158)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory containing processed patches')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size used in original extraction')
    parser.add_argument('--stride', type=int, default=128,
                       help='Stride used in original extraction')
    
    args = parser.parse_args()

    if args.sample_id is None:
        args.sample_id = [d for d in os.listdir(args.output_dir) 
                      if os.path.isdir(os.path.join(args.output_dir, d))]
        logger.info(f"No sample_id provided. Found samples: {args.sample_id}")
        # sys.exit(1)

    if len(args.sample_id) > 1:
        logger.info(f"Processing multiple samples: {args.sample_id}")
        num_cpus = 4
        jobs = []

        # use apply async to have more control and logging
        pool = multiprocessing.Pool(processes=num_cpus)
        for sid in args.sample_id:
            logger.info(f"Starting coordinate addition for sample: {sid}")
            # use apply async!!!
            job = pool.apply_async(
                add_coordinates_to_existing_metadata,
                (args.data_dir, sid, args.output_dir, args.patch_size, args.stride)
            )
            jobs.append((sid, job))
        pool.close()
        pool.join()
        for sid, job in jobs:
            result = job.get()
            if not isinstance(result, Exception):
                logger.info(f"Completed coordinate addition for sample: {sid}")
            else:
                logger.error(f"Failed coordinate addition for sample: {sid} ({result})")
        logger.info("Coordinate addition complete for all samples!")
        sys.exit(0)
    else:
        # Add coordinates to existing metadata
        add_coordinates_to_existing_metadata(
            data_dir=args.data_dir,
            sample_id=args.sample_id,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride
        )
        
        logger.info("Coordinate addition complete!")
