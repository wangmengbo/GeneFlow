import os, h5py, json, tifffile, cv2, argparse, random, gc, multiprocessing, pickle, logging
# import torch
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from skimage.measure import regionprops
from tqdm import tqdm
from pathlib import Path
from functools import partial
from scipy.ndimage import binary_erosion
from concurrent.futures import ProcessPoolExecutor
from shapely import box
from shapely.geometry import Polygon
from shapely.affinity import translate
import spatialdata_io
import geopandas as gpd
import swifter
import multiprocessing as mp
from PIL import Image
from skimage import transform
from joblib import Parallel, delayed
from gene_thesaurus import GeneThesaurus
from adjustText import adjust_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

debug = False
save_tmp = False


def extract_coordinates(polygon):
    coords = list(polygon.exterior.coords)
    x_coords = [point[0] for point in coords]
    y_coords = [point[1] for point in coords]
    return pd.DataFrame({'x':x_coords, 'y':y_coords})


def extract_centroid(polygon):
    coords = polygon.centroid.xy
    x = coords[0][0]
    y = coords[1][0]
    return pd.DataFrame({'x':[x], 'y':[y]})


def apply_transformation(coords, inv_matrix, scaling_factor=0.2125):
    """Vectorized coordinate transformation with precomputed inverse matrix"""
    scaled_coords = coords / scaling_factor
    homogeneous = np.hstack([scaled_coords, np.ones((len(scaled_coords), 1))])
    return homogeneous @ inv_matrix.T[:, :2]


def normalize_image(img, convert_to=np.uint8):
    """Convert image to 0-255 range"""
    global debug

    if img.dtype != np.uint8:
        if (np.issubdtype(img.dtype, np.floating) and img.max() > 1) or img.max() > 255:
                img = (img + 1e-6) / (img.max() - img.min() + 1e-6) * 255

    if convert_to is not None:
        logger.debug(f"Converting image dtype: {img.dtype} -> {convert_to}")
        img = img.astype(convert_to)

    return img


def transform_image(img, target_shape, alignment_matrix):
    return transform.warp(
        img,
        transform.AffineTransform(matrix=alignment_matrix),
        output_shape=target_shape,
        preserve_range=True
    )


def transform_polygon(polygon, inv_matrix, scaling_factor=0.2125):
    """Batch-process all polygon coordinates"""
    exterior = apply_transformation(np.array(polygon.exterior.coords), inv_matrix, scaling_factor=scaling_factor)
    interiors = [apply_transformation(np.array(i.coords), inv_matrix, scaling_factor=scaling_factor) for i in polygon.interiors]
    return Polygon(exterior, interiors)


def parallel_transform_polygons(gdf, inv_matrix, scaling_factor=0.2125):
    """Parallelized transformation with matrix precomputation"""
    return pd.DataFrame(gdf.swifter.apply(
        lambda row: transform_polygon(row.geometry, inv_matrix, scaling_factor=scaling_factor),
        axis=1
    )).rename({0: 'geometry'}, axis=1)


def plot_image(img, show=True, mask_channel=None):
    rgb = False
    if img.shape[2] < 3:
        n = img.shape[2]
    else:
        rgb = True
        n = 1 + img.shape[2]-3

    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        ax = axes[0, i]
        if rgb:
            if i==0:
                if mask_channel is not None:
                    img_w_mask = cv2.bitwise_and(img[:,:,:3], img[:,:,mask_channel])
                    ax.imshow(img_w_mask)
                else:
                    ax.imshow(img[:,:,:3])
            else:
                ax.imshow(img[:,:,i+2])
        else:
            ax.imshow(img[:,:,i])
        ax.axis('off')
    if show:
        plt.show()
    else:
        return fig

def plot_total_counts(adata, 
                      cnt_threshold: list[float,int]=None, 
                      pct_threshold: list[float,int]=None,
                      figsize=(5, 5),
                      show=True):
    fig, ax = plt.subplots(figsize=figsize)
    sns.ecdfplot(data=adata.obs['total_counts'], ax=ax)
    thresholds = []
    threshold_labels = []
    if cnt_threshold is not None:
        thresholds = cnt_threshold
        threshold_labels = [None for i in cnt_threshold]
    
    if pct_threshold is not None:
        for i in pct_threshold:
            t = np.percentile(adata.obs['total_counts'], i)
            thresholds.append(t)
        threshold_labels.extend(pct_threshold)

    logger.info(f"thresholds: {thresholds}")
    
    # Create a list to store all text objects
    texts = []
    
    for t, tl in zip(thresholds, threshold_labels):
        proportion = np.sum(adata.obs['total_counts'] < t) / adata.n_obs
        ax.axvline(x=t, ymin=0, ymax=proportion, color='red', linewidth=.5, linestyle='--')
        
        # Add text and store the text object
        text_obj = ax.text(t, 0, f"{t:.0f}\n({tl}%)" if tl is not None else f"{t:.0f}", 
                           color='red', ha='center', va='bottom')
        texts.append(text_obj)
        
        x_range = ax.get_xlim()
        xmax_axes = (t - x_range[0]) / (x_range[1] - x_range[0])
        ax.axhline(y=proportion, xmin=0, xmax=xmax_axes, color='red', linewidth=.5, linestyle='--')
        
        # Add text and store the text object
        text_obj = ax.text(-.2, proportion, f"{proportion:.2f}", color='red', ha='right', va='bottom')
        # texts.append(text_obj)
    
    # Apply adjustText to avoid overlaps
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                expand_points=(1.5, 1.5),
                force_points=0.1)
    
    if show:
        plt.show()
    return fig, ax


def load_xenium_data(adata=None,
    img=None,
    input_path=None,
    sdata=None,
    cell_feature_matrix_path=None,
    cell_meta=None,
    cell_summary_path=None,
    histology_image_path=None,
    spatial_coords_names=["x_centroid", "y_centroid"],
    do_scaling=True,
    scaling_factor=0.2125,
    alignment_matrix=None,
    alignment_matrix_delimiter=',',
    cell_boundaries=None,  # transformed cell boundaries
    nucleus_boundaries=None,  # transformed cell boundaries
    clip_extra_channels={},
    clip_pct_threshold=99,
    replace_with_reference_gene_names=True,
    min_counts=0,
    preprocess=True,
    ) -> sc.AnnData:
    if input_path is not None:
        sdata = spatialdata_io.xenium(input_path)

    if sdata is not None:
        assert 'he_image' in sdata.images, "No histological image found in spatialdata object"

        img = np.transpose(sdata['he_image']['scale0'].data_vars['image'].to_numpy(), (1,2,0))
        adata = sdata['table']
        adata.obs_names = adata.obs['cell_id']
        adata.uns['he_image'] = np.ascontiguousarray(img)
        
        transformed_coords = adata.obsm["spatial"].copy()

        if scaling_factor is not None:
            transformed_coords /= scaling_factor  # Convert microns to pixels

        alignment_matrix = sdata.images['he_image']['/scale0'].data_vars['image'].transform['global'].matrix
        inv_alignment_matrix = np.linalg.inv(alignment_matrix)

        transformed_coords = apply_transformation(transformed_coords, inv_alignment_matrix)
        
        # Also store original coordinates
        adata.obsm["spatial_microns"] = adata.obsm["spatial"]
        adata.obsm["spatial"] = transformed_coords

        if cell_boundaries is None:
            cell_boundaries = parallel_transform_polygons(sdata['cell_boundaries'], 
                                                                      inv_alignment_matrix, 
                                                                      scaling_factor=scaling_factor)
        shared_obs_names = list(set(adata.obs_names).intersection(set(cell_boundaries.index)))
        adata = adata[shared_obs_names, :]
        adata.obsm["cell_boundaries"] = cell_boundaries.loc[adata.obs_names,:]

        if nucleus_boundaries is None:
            nucleus_boundaries = parallel_transform_polygons(sdata['nucleus_boundaries'],
                                                                         inv_alignment_matrix, 
                                                                         scaling_factor=scaling_factor)
        shared_obs_names = list(set(adata.obs_names).intersection(set(nucleus_boundaries.index)))
        adata = adata[shared_obs_names, :]
        adata.obsm["nucleus_boundaries"] = nucleus_boundaries.loc[adata.obs_names,:]

        for k,v in sdata.images.items():
            if k == 'he_image':
                continue
            aligned_img = np.transpose(v['scale0'].data_vars['image'].to_numpy(), (1,2,0))

            if aligned_img.shape[2] > 4:
                aligned_img = aligned_img[:,:,:4]

            if k in clip_extra_channels:
                if clip_extra_channels[k] is None:
                    thresholds = np.percentile(aligned_img, clip_pct_threshold, axis=(0,1))
                else:
                    thresholds = clip_extra_channels[k]
                for i,threshold in enumerate(thresholds):
                    aligned_img_tmp = aligned_img[:,:,i]
                    logger.info(f"Clipping {k}, {i} to (0, {threshold})")
                    aligned_img_tmp = np.clip(aligned_img_tmp, 0, threshold)
                    aligned_img_tmp = normalize_image(aligned_img_tmp, convert_to=np.uint8)
                    aligned_img[:, :, i] = aligned_img_tmp
            aligned_img = transform_image(aligned_img, img.shape[:2], alignment_matrix)
            for i in range(aligned_img.shape[2]):
                aligned_img[:,:,i] = normalize_image(aligned_img[:,:,i], convert_to=np.uint8)

            adata.uns[k] = np.ascontiguousarray(aligned_img)
    else:
        if adata is None:
            adata = sc.read_10x_h5(cell_feature_matrix_path)
            
        if cell_meta is None:
            # Load cell metadata
            cell_meta = pd.read_csv(cell_summary_path)
        
        # Ensure cell_id is used as index
        if "cell_id" in cell_meta.columns:
            cell_meta = cell_meta.set_index("cell_id")
        
        # Add cell metadata to AnnData
        for col in cell_meta.columns:
            adata.obs[col] = cell_meta[col].values
        
        # Load histological image if not provided
        if img is None and histology_image_path is not None:
            logger.info("Loading histological image...")
            # Load image with tifffile to access metadata
            with tifffile.TiffFile(histology_image_path) as tif:
                img = tif.asarray()
        
        adata.uns["he_image"] = img
        
        # Add spatial coordinates to obsm with proper transformation
        if all(coord in cell_meta.columns for coord in spatial_coords_names):
            # Extract coordinates from metadata (in microns)
            spatial_coords = cell_meta[spatial_coords_names].values
            
            # If we have the image and scale factors, transform coordinates
            if img is not None and scaling_factor is not None:
                transformed_coords = spatial_coords.copy()

                # Convert from microns to pixels
                if do_scaling:
                    transformed_coords /= scaling_factor  # Convert microns to pixels

                if alignment_matrix is not None:
                    if type(alignment_matrix) is str:
                        alignment_matrix = np.loadtxt(alignment_matrix, delimiter=alignment_matrix_delimiter)
                    transformed_coords = apply_transformation(transformed_coords, alignment_matrix)
                
                # Store transformed coordinates
                adata.obsm["spatial"] = transformed_coords
                
                # Also store original coordinates
                adata.obsm["spatial_microns"] = spatial_coords
            else:
                # If no image or scale factors, just store the original coordinates
                adata.obsm["spatial"] = spatial_coords

    recount_total = False
    # Filter features to include only gene expression
    if 'feature_type' in adata.var.columns and adata.var.feature_type.nunique() > 1:
        logger.warning(f"Filtering features to include only gene expression: {adata.var.feature_type.unique()}")
        logger.warning(f"Remove non-gene expression features: {adata.var.feature_type.unique()}")
        adata = adata[:, adata.var['feature_type'] == 'Gene Expression'].copy()
        recount_total = True
        gc.collect()

    # Calculate total counts per cell and add to adata.obs
    if 'total_counts' not in adata.obs or recount_total:
        adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
    
    threshold_low = np.percentile(adata.obs['total_counts'], 5)
    threshold_high = np.percentile(adata.obs['total_counts'], 95)
    adata.obs['low_quality'] = (adata.obs['total_counts'] < threshold_low) | (adata.obs['total_counts'] > threshold_high)
    adata.uns['total_counts_quantiles'] = {'q05': threshold_low, 'q95': threshold_high}
    logger.info(adata.obs['low_quality'].value_counts())
    
    plot_total_counts(adata, pct_threshold=[5, 95], cnt_threshold=[100])
    
    if replace_with_reference_gene_names:
        gene_symbols = pd.read_csv("gene_symbols.txt", header=None)[0].tolist()
        new_var_names = []
        gene_symbol_alt = {
            'HIST1H4C': 'H4C3',
            'LOR': 'LORICRIN',
            "ACPP": "ACP3",
            "ADGRD2": "GPR144",
            "EPRS": "GLUPRORS",
            "H2AFX": "H2AX",
            "H3F3B": "H3-3B",
            "IGHE": "IgE",
            "TMEM173": "STING1",
            "TRDC": "TCRD",
            "WARS": "WARS1",
            "CBSL": 'LOC102724560',
        }
        os.makedirs("gene_thesaurus", exist_ok=True)
        gt = GeneThesaurus(data_dir='gene_thesaurus')
        alt_not_found = []
        alt_found = {}
        for i in adata.var_names:
            if i not in gene_symbols:
                rescued_alt = gt.update_gene_symbols([i])
                if i in gene_symbol_alt and gene_symbol_alt[i] in gene_symbols:
                    alt_found[i] = gene_symbol_alt[i]
                    new_var_names.append(gene_symbol_alt[i])
                elif len(rescued_alt) > 0 and rescued_alt[i] in gene_symbols:
                    # print(rescued_alt)
                    alt_found[i] = rescued_alt[i]
                    new_var_names.append(rescued_alt[i])
                else:
                    alt_not_found.append(i)
                    new_var_names.append(i)
            else:
                new_var_names.append(i)
        logger.info(f"Total {len(alt_not_found) + len(alt_not_found)} misspecified gene names: {len(alt_found)} found alternative ({alt_found}), {len(alt_not_found)} not found ({alt_not_found})")
        adata.var_names = new_var_names

    if min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata)
    if preprocess:
        sc.tl.pca(adata, random_state=42)
        sc.pp.neighbors(adata, random_state=42)
        sc.tl.umap(adata, random_state=42)
        sc.tl.leiden(adata, random_state=42)
        sc.tl.rank_genes_groups(adata, groupby="leiden", random_state=42)

    return adata


def visualize_cell_coordinates_on_image(
    img,
    adata=None,
    coordinates=None,
    output_path=None,
    matched_color=(0, 255, 0),  # Green for matched cells
    unmatched_color=(1, 0, 0),  # Red for unmatched cells
    centroid_radius=5,
    line_thickness=2,
    draw_connections=True,
    draw_unmatched=True,
    alpha=0.7
    ):
    """
    Visualize cell coordinates on the entire H&E image without segmentation.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression data with spatial coordinates
    img : numpy.ndarray or str
        Histological image (3D array with color channels) or path to image file
    coordinates : pandas.DataFrame, optional
        DataFrame containing cell coordinates if not in adata.obsm["spatial"]
    output_path : str, optional
        Path to save the output visualization image
    matched_color : tuple, optional
        RGB color for matched cells (0-255 scale)
    unmatched_color : tuple, optional
        RGB color for unmatched cells (0-255 scale)
    centroid_radius : int, optional
        Radius of circles drawn at cell centroids
    line_thickness : int, optional
        Thickness of lines connecting points
    draw_connections : bool, optional
        Whether to draw lines between matched coordinates
    draw_unmatched : bool, optional
        Whether to visualize unmatched cells
    alpha : float, optional
        Transparency of the overlay (0.0 to 1.0)
        
    Returns
    -------
    vis_img : numpy.ndarray
        Visualization image with cell coordinates drawn
    """
    
    # Load image if path is provided
    if isinstance(img, str):
        img = tifffile.imread(img)
    
    # Get cell coordinates from AnnData or provided coordinates
    if coordinates is not None:
        cell_coords = coordinates.values
        cell_ids = coordinates.index.tolist()
    elif "spatial" in adata.obsm:
        cell_coords = adata.obsm["spatial"]
        cell_ids = adata.obs_names.tolist()
    else:
        raise ValueError("Cell coordinates not found. Please provide coordinates or ensure they exist in adata.obsm['spatial']")
    
    logger.info(f"Total cells to visualize: {len(cell_ids)}")
    
    # Create a copy of the image for visualization
    # vis_img = img.copy()
    vis_img = img
    
    # Make sure the image has 3 channels for RGB visualization
    if len(vis_img.shape) == 2 or (len(vis_img.shape) == 3 and vis_img.shape[2] == 1):
        vis_img = np.repeat(vis_img[:, :, np.newaxis] if len(vis_img.shape) == 2 else vis_img, 3, axis=2)
    
    # Convert to float32 for visualization if needed
    if vis_img.dtype != np.float32:
        vis_img = vis_img.astype(np.float32) / 255.0 if vis_img.max() > 1.0 else vis_img.astype(np.float32)
    logger.info("vis_img format", vis_img.min(), vis_img.max(), vis_img.dtype)

    # Normalize colors to 0-1 range for float32 image
    matched_color_norm, unmatched_color_norm = matched_color, unmatched_color
    if max(matched_color_norm) > 1.0:
        matched_color_norm = (matched_color_norm[0]/255.0, matched_color_norm[1]/255.0, matched_color_norm[2]/255.0)
    if max(unmatched_color_norm) > 1.0:
        unmatched_color_norm = (unmatched_color_norm[0]/255.0, unmatched_color_norm[1]/255.0, unmatched_color_norm[2]/255.0)
    logger.info(f"matched_color_norm={matched_color_norm}; unmatched_color_norm={unmatched_color_norm}")

    # Draw cell centroids
    height, width = vis_img.shape[:2]
    
    # Create a set of cells that have been matched (if there's a way to determine this)
    # For this example, we'll assume all cells in adata are matched
    matched_cells = set(cell_ids)
    
    # Draw all cells
    for i, (x, y) in tqdm(enumerate(cell_coords), total=len(cell_coords), desc="Drawing cells"):
        cell_id = cell_ids[i]
        
        # Skip cells outside the image boundaries
        if not (0 <= y < height and 0 <= x < width):
            continue
        
        # Determine if the cell is matched or unmatched
        if cell_id in matched_cells:
            color = matched_color_norm
        else:
            if not draw_unmatched:
                continue
            color = unmatched_color_norm
        
        # Draw cell centroid as a circle
        # logger.info(vis_img.shape, int(x), int(y))
        cv2.circle(vis_img, (int(x), int(y)), centroid_radius, color, -1)  # -1 means filled circle
    
    # Convert back to uint8 for saving
    if output_path is not None:
        save_img = (vis_img * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tifffile.imwrite(output_path, save_img)
        logger.info(f"Visualization saved to {output_path}")
    
    return vis_img


def create_cell_images(cell_img, cell_mask, local_cell_poly, cell_boundary_color=(0, 255, 0), 
                       nucleus_mask=None, local_nucleus_poly=None, nucleus_boundary_color=(255, 0, 0)):
    global debug

    # Create masked version
    masked_cell_img = cv2.bitwise_and(cell_img, cell_img, mask=cell_mask)

    # Draw boundary with improved visibility
    boundary_cell_img = cell_img[:,:,:3].copy()

    int_coords = np.array(local_cell_poly.exterior.coords.xy).T.astype(int)
    # Use a higher contrast color and thicker line
    cv2.polylines(boundary_cell_img, [int_coords], isClosed=True, color=cell_boundary_color, thickness=4)
    
    # Add debug check to verify polygon is visible
    if np.all(boundary_cell_img == cell_img[:,:,:3]):
        logger.info(f"Warning: Boundary drawing had no effect. Polygon might be outside image.")
        # Force draw at center as fallback
        center_x, center_y = cell_img.shape[1]//2, cell_img.shape[0]//2
        cv2.circle(boundary_cell_img, (center_x, center_y), 10, cell_boundary_color, -1)

    # Draw nucleus boundary if provided
    if nucleus_mask is not None and local_nucleus_poly is not None:
        int_nucleus_coords = np.array(local_nucleus_poly.exterior.coords.xy).T.astype(int)
        cv2.polylines(boundary_cell_img, [int_nucleus_coords], isClosed=True, color=nucleus_boundary_color, thickness=4)
    
    unmasked_img = np.concatenate([cell_img.copy(), 
                                   cell_mask.reshape((cell_mask.shape[0], cell_mask.shape[1], 1)), 
                                   nucleus_mask.reshape((nucleus_mask.shape[0], nucleus_mask.shape[1], 1))], axis=2)
    
    return {
        'unmasked': unmasked_img,
        'masked': masked_cell_img,
        'boundary': boundary_cell_img,
        'cell_polygon': local_cell_poly,
        'nucleus_polygon': local_nucleus_poly,
    }


def extract_and_pad_cell(img, x_min, x_max, y_min, y_max, size, cell_polygon, nucleus_polygon=None):
    """Extract cell image with padding and create mask"""
    global debug
    if debug:
        logger.info(f"img.shape={img.shape}")
        
    # Calculate padding needs
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - img.shape[1])
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - img.shape[0])
    
    if debug:
        logger.info(f"Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")

    # Extract visible portion
    visible = img[
        max(y_min, 0):min(y_max, img.shape[0]),
        max(x_min, 0):min(x_max, img.shape[1])
    ]
    
    if debug:
        plt.imshow(visible)
        plt.title("Visible Portion of Cell Image")
        plt.show()

    # Create padded array
    cell_img = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    cell_img[
        pad_top : pad_top+visible.shape[0],
        pad_left : pad_left+visible.shape[1]
    ] = visible
    # logger.info(f"[extract_and_pad_cell] np.unique(cell_img): {np.unique(cell_img)}")

    if debug:
        plt.imshow(cell_img)
        plt.title("Padded Cell Image")
        plt.show()

    masks = {'cell_mask': None, 'local_cell_poly': None, 'nucleus_mask': None, 'local_nucleus_poly': None}

    # Create mask with validation
    for k, polygon in zip(['cell', 'nucleus'], [cell_polygon, nucleus_polygon]):
        if polygon is None:
            continue

        mask = np.zeros((size, size), dtype=np.uint8)
        if debug:
            logger.info(f"Translating polygon to local coordinates: xoff={-x_min+pad_left}, yoff={-y_min+pad_top}")
        local_poly = translate(polygon, xoff=-x_min+pad_left, yoff=-y_min+pad_top)
        poly_points = np.array(local_poly.exterior.coords.xy).T.astype(int)
        if poly_points.shape[0] > 2:  # Need at least 3 points
            cv2.fillPoly(mask, [poly_points], 255)
            # Debug check
            if np.sum(mask) == 0:
                logger.info(f"Warning: Empty mask created. Polygon might be outside image.")
        else:
            logger.info(f"Warning: Invalid polygon with {poly_points.shape[0]} points")
        
        if debug:
            plt.imshow(mask)
            plt.title("Mask for Cell Image")
            plt.show()
        
        masks[f"{k}_mask"] = mask
        masks[f"local_{k}_poly"] = local_poly

    return cell_img, masks['cell_mask'], masks['local_cell_poly'], masks['nucleus_mask'], masks['local_nucleus_poly']


def count_patches(
    img=None,
    height=None,
    width=None,
    window_size=512,
    overlap=100,
):
    global debug

    height, width, _ = img.shape

    # Calculate sliding window positions
    step_size = window_size - overlap
    windows_y = range(0, height - window_size + 1, step_size)
    windows_x = range(0, width - window_size + 1, step_size)
    windows_y = sorted(set(list(windows_y) + [height - window_size]))
    windows_x = sorted(set(list(windows_x) + [width - window_size]))
    
    all_patches = []
    for i, y in enumerate(windows_y):
        for j, x in enumerate(windows_x):
            window_id = f"window_{i}_{j}"
            all_patches.append((y, x, window_id))
    logger.info(f"Image size: {height}x{width}, Window size: {window_size}, Overlap: {overlap}")
    logger.info(f"Total windows: {len(all_patches)}")
    return all_patches


def load_xenium_data_with_polygons_single_thread(
    adata,
    img=None,
    extra_img_keys=None,
    cell_polygons_df=None,
    nucleus_polygons_df=None,
    output_dir=None,
    window_size=512,
    overlap=100,
    cell_square_size=512,
    debug_mode=False,
    selected_patches=None,
    num_debug_patches=1,
    random_seed=42,
    cell_boundary_color=(0, 255, 0),  # Green in RGB
    nucleus_boundary_color=(255, 0, 0),  # Red in RGB
    skip_empty_patch=False,
    extract_cell_image=True,
    extract_patch_image=True,
    verbose=False,
    return_images=False,
    overwrite=True,
):
    """
    Process histological image using sliding window and predefined cell polygons.
    
    Parameters
    ----------
    cell_polygons_df : pd.DataFrame
        DataFrame containing cell polygons with cell_id as index
    boundary_color : tuple, optional
        RGB color for cell boundaries (0-1 range)
    """
    global debug

    # Load image and prepare data structures
    if img is None:
        if 'he_image' in adata.uns:
            img = adata.uns['he_image']

        imgs = [img]
        if extra_img_keys is not None:
            for k in extra_img_keys:
                if k in adata.uns:
                    imgs.append(adata.uns[k])

        # combine img_extra and img on the 3rd dimension
        if len(imgs) > 0:
            img = np.concatenate(imgs, axis=2)
    elif type(img) is str:
        img = tifffile.imread(img)

    # convert to 0-255
    img = normalize_image(img, convert_to_uint8=True)

    height, width, _ = img.shape
    
    if cell_polygons_df is None:
        cell_polygons_df = adata.obsm['cell_boundaries']

    if nucleus_polygons_df is None:
        nucleus_polygons_df = adata.obsm['nucleus_boundaries']

    polygons_df = pd.merge(cell_polygons_df, nucleus_polygons_df, left_index=True, right_index=True, suffixes=('_cell', '_nucleus'))

    # Use existing Polygon objects directly
    cell_geometries = {
        cell_id: {
            'cell_polygon': row['geometry_cell'],
            'cell_bounds': row['geometry_cell'].bounds,
            'nucleus_polygon': row['geometry_nucleus'],
            'nucleus_bounds': row['geometry_nucleus'].bounds,
        }
        for cell_id, row in polygons_df.iterrows()
    }

    # Calculate sliding window positions
    step_size = window_size - overlap
    windows_y = range(0, height - window_size + 1, step_size)
    windows_x = range(0, width - window_size + 1, step_size)
    windows_y = sorted(set(list(windows_y) + [height - window_size]))
    windows_x = sorted(set(list(windows_x) + [width - window_size]))
    
    # Create a list of all patches
    if selected_patches is None:
        all_patches = []
        for i, y in enumerate(windows_y):
            for j, x in enumerate(windows_x):
                window_id = f"window_{i}_{j}"
                all_patches.append((y, x, window_id))
        logger.info(f"Image size: {height}x{width}, Window size: {window_size}, Overlap: {overlap}")
        logger.info(f"Total windows: {len(all_patches)}")
        if debug_mode:
            random.seed(random_seed)  # For reproducibility
            selected_patches = random.sample(all_patches, min(num_debug_patches, len(all_patches)))
            logger.info(f"Processing {len(selected_patches)} random patches")
        else:
            selected_patches = all_patches
    logger.info(f"Processing {len(selected_patches)} patches")

    window_cell_dict = {}

    adata.obs['window_id'] = pd.NA

    cell_images = {}
    patch_images = []
    
    # Process each window
    for y, x, window_id in tqdm(selected_patches, desc="Processing patches"):
        if not return_images:
            cell_images = {}
            patch_images = []

        window_bounds = box(x, y, x+window_size, y+window_size)
        window_img = img[y:y+window_size, x:x+window_size]
        
        # Find cells intersecting this window using spatial predicate
        cells_in_window = [
            (cell_id, geo['cell_polygon'], geo['nucleus_polygon']) 
            for cell_id, geo in cell_geometries.items()
            if geo['cell_polygon'].intersects(window_bounds)
        ]
        if verbose:
            logger.info(f"Found {len(cells_in_window)} cells in window ({y}, {x}, {window_id}).")
        
        if len(cells_in_window)==0 and skip_empty_patch:
            if verbose:
                logger.info(f"Window ({y}, {x}, {window_id}) is empty, skipping...")
            continue
        
        cell_ids_tmp = set([cell_id for (cell_id, _, _) in cells_in_window])
        if window_id in window_cell_dict:
            window_cell_dict[window_id] = window_cell_dict[window_id].update(cell_ids_tmp)
        else:
            window_cell_dict[window_id] = cell_ids_tmp
        
        # Update AnnData
        adata.obs.loc[adata.obs_names.isin([i[0] for i in cells_in_window]), 'window_id'] = window_id

        # Create visualization images
        if extract_cell_image or extract_patch_image:
            
            if output_dir is not None and extract_patch_image:
                patch_dir = os.path.join(output_dir, "patch_images")
                p_patch_orignal_img = os.path.join(patch_dir, f"{window_id}_original.tif")
                p_patch_boundary_img = os.path.join(patch_dir, f"{window_id}_boundary.tif")
                if not overwrite and \
                    os.path.exists(p_patch_orignal_img) and \
                        os.path.exists(p_patch_boundary_img):
                    if verbose:
                        logger.info(f"Patch {window_id} already exists, skipping...")
                    continue

            ncells = 0
            patch_cell_mask = np.zeros(window_img.shape[:2], dtype=np.uint8)
            patch_nucleus_mask = np.zeros(window_img.shape[:2], dtype=np.uint8)
            patch_image_w_boundary = window_img[:, :, :3].copy() # only draw boundary on RGB channels
            
            if output_dir is not None and extract_cell_image:
                cell_dir = os.path.join(output_dir, "cell_images")
            
            for cell_id, cell_polygon, nucleus_polygon in cells_in_window:
                if output_dir is not None and extract_cell_image:
                    p_cell_unmasked = os.path.join(cell_dir, f"{cell_id}_original.tif")
                    p_cell_masked = os.path.join(cell_dir, f"{cell_id}_masked.tif")
                    p_cell_boundary = os.path.join(cell_dir, f"{cell_id}_boundary.tif")
                    if not overwrite and \
                        os.path.exists(p_cell_unmasked) and \
                            os.path.exists(p_cell_masked) and \
                                os.path.exists(p_cell_boundary):
                        if verbose:
                            logger.info(f"Cell {cell_id} already exists, skipping...")
                        continue

                # Convert global polygon to window coordinates
                local_cell_poly = translate(cell_polygon, xoff=-x, yoff=-y)
                int_cell_coords = np.array(local_cell_poly.exterior.coords.xy).T.astype(int)
                local_nucleus_poly = translate(nucleus_polygon, xoff=-x, yoff=-y)
                int_nucleus_coords = np.array(local_nucleus_poly.exterior.coords.xy).T.astype(int)
                
                # Create patch mask for cell and nucleus, and draw boundaries
                if int_cell_coords.shape[0] > 2:  # Need at least 3 points
                    cv2.fillPoly(patch_cell_mask, [int_cell_coords], 255)
                    cv2.polylines(patch_image_w_boundary, [int_cell_coords], 
                            isClosed=True, color=cell_boundary_color, thickness=2)

                if int_nucleus_coords.shape[0] > 2:  # Need at least 3 points
                    cv2.fillPoly(patch_nucleus_mask, [int_nucleus_coords], 255)
                    cv2.polylines(patch_image_w_boundary, [int_nucleus_coords],
                            isClosed=True, color=nucleus_boundary_color, thickness=2)              

                # Extract cell image
                if extract_cell_image:
                    if cell_id not in cell_images:
                        centroid = cell_polygon.centroid.coords[0]
                        
                        # Calculate square bounds around centroid
                        half_size = cell_square_size // 2
                        y_min = int(centroid[1] - half_size)
                        y_max = int(centroid[1] + half_size)
                        x_min = int(centroid[0] - half_size)
                        x_max = int(centroid[0] + half_size)

                        # Extract and pad image
                        if debug:
                            logger.info(f"Extracting cell {cell_id} at ({x_min}, {x_max}, {y_min}, {y_max})")
                        
                        cell_img, cell_mask, local_cell_poly, nucleus_mask, local_nucleus_poly = extract_and_pad_cell(
                            img, x_min, x_max, y_min, y_max, 
                            cell_square_size, cell_polygon, 
                            nucleus_polygon=nucleus_polygon
                        )
                        # logger.info(f"[load_xenium_data_with_polygons_single_thread] np.unique(cell_img): {np.unique(cell_img)}")

                        cell_image_tmp = create_cell_images(
                            cell_img, cell_mask, local_cell_poly,
                            cell_boundary_color=cell_boundary_color,
                            nucleus_mask=nucleus_mask,
                            local_nucleus_poly=local_nucleus_poly,
                            nucleus_boundary_color=nucleus_boundary_color,
                        )
                        cell_image_tmp['window_id'] = window_id
                        
                        if output_dir is not None:
                            cell_image_tmp['p_cell_unmasked'] = p_cell_unmasked
                            cell_image_tmp['p_cell_masked'] = p_cell_masked
                            cell_image_tmp['p_cell_boundary'] = p_cell_boundary

                        cell_images[cell_id] = cell_image_tmp
                        ncells += 1

            if verbose:
                logger.info(f"Extracted {ncells} cells in window ({y}, {x}, {window_id}).")

            patch_unmasked_img = np.concatenate([window_img, 
                                                 patch_cell_mask.reshape((patch_cell_mask.shape[0], patch_cell_mask.shape[1], 1)),
                                                 patch_nucleus_mask.reshape((patch_nucleus_mask.shape[0], patch_nucleus_mask.shape[1], 1))], axis=2)
            patch_masked_img = cv2.bitwise_and(window_img, window_img, mask=patch_cell_mask)

            # Store patch data
            patch_image_tmp = {
                'window_id': window_id,
                'unmasked': patch_unmasked_img,
                'masked': patch_masked_img,
                'boundary': patch_image_w_boundary,
                'cells': [cid for (cid, _, _) in cells_in_window],
            }
            if output_dir is not None and extract_patch_image:
                patch_image_tmp['p_patch_unmasked_img'] = p_patch_orignal_img
                patch_image_tmp['p_patch_masked_img'] = p_patch_orignal_img
                patch_image_tmp['p_patch_boundary_img'] = p_patch_boundary_img

            patch_images.append(patch_image_tmp)

            # Save outputs
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

                if extract_cell_image:
                    # Save cell images
                    os.makedirs(cell_dir, exist_ok=True)
                    for cell_id, imgs in cell_images.items():
                        tifffile.imwrite(imgs['p_cell_unmasked'], imgs['unmasked'])
                        # tifffile.imwrite(imgs['p_cell_masked'], imgs['masked'])
                        # tifffile.imwrite(imgs['p_cell_boundary'], imgs['boundary'])
                
                if extract_patch_image:
                    # Save patch images
                    os.makedirs(patch_dir, exist_ok=True)
                    for patch in patch_images:
                        tifffile.imwrite(patch['p_patch_unmasked_img'], patch['unmasked'])
                        # tifffile.imwrite(patch['p_patch_masked_img'], patch['masked'])
                        # tifffile.imwrite(patch['p_patch_boundary_img'], patch['boundary'])

        gc.collect()
    
    return adata, cell_images, patch_images, window_cell_dict



def load_xenium_data_with_polygons_parallel(
    adata,
    img=None,
    extra_img_keys=None,
    cell_polygons_df=None,
    nucleus_polygons_df=None,
    output_dir=None,
    patch_size=256,
    overlap=128,
    cell_size=256,
    test_mode=False,
    selected_patches=None,
    npatches=1,
    random_seed=42,
    cell_boundary_color=(0, 255, 0),  # Green in RGB
    nucleus_boundary_color=(255, 0, 0),  # Red in RGB
    skip_empty_patch=False,
    extract_cell_image=True,
    extract_patch_image=True,
    return_images=False,
    verbose=False,
    save_all_images=False,
    overwrite=False,
    n_jobs=-1,  # Number of parallel jobs, -1 means use all available cores
    batch_size=100,  # Number of patches per parallel job
):
    """
    Process histological image using sliding window and predefined cell polygons with parallel processing.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing cell data
    img : numpy.ndarray or str, optional
        Image data or path to image file
    extra_img_keys : list, optional
        List of keys for additional images in adata.uns
    cell_polygons_df : pd.DataFrame, optional
        DataFrame containing cell polygons with cell_id as index
    nucleus_polygons_df : pd.DataFrame, optional
        DataFrame containing nucleus polygons with cell_id as index
    output_dir : str, optional
        Directory to save output images
    patch_size : int, default=256
        Size of sliding window
    overlap : int, default=128
        Overlap between adjacent windows
    cell_size : int, default=256
        Size of square around cell centroid
    test_mode : bool, default=False
        Whether to run in test mode (random subset)
    selected_patches : list, optional
        List of patches to process
    npatches : int, default=1
        Number of patches to process in test mode
    random_seed : int, default=42
        Random seed for reproducibility
    cell_boundary_color : tuple, default=(0, 255, 0)
        RGB color for cell boundaries
    nucleus_boundary_color : tuple, default=(255, 0, 0)
        RGB color for nucleus boundaries
    skip_empty_patch : bool, default=False
        Whether to skip empty patches
    extract_cell_image : bool, default=True
        Whether to extract cell images
    extract_patch_image : bool, default=True
        Whether to extract patch images
    return_images : bool, default=False
        Whether to return images
    verbose : bool, default=False
        Whether to print verbose output
    save_all_images : bool, default=False
        Whether to save all images
    overwrite : bool, default=False
        Whether to overwrite existing files
    n_jobs : int, default=-1
        Number of parallel jobs, -1 means use all available cores
    batch_size : int, default=100
        Number of patches per parallel job
    """
    import os
    import numpy as np
    import tifffile
    import random
    import multiprocessing
    import pandas as pd
    import gc
    import logging
    from shapely.geometry import box
    from shapely.affinity import translate
    from tqdm import tqdm
    import cv2
    from joblib import Parallel, delayed, parallel_backend

    global debug

    # Determine number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Load image and prepare data structures
    if img is None:
        if 'he_image' in adata.uns:
            img = adata.uns['he_image']
        imgs = [img]
        if extra_img_keys is not None:
            for k in extra_img_keys:
                if k in adata.uns:
                    imgs.append(adata.uns[k])
        # combine img_extra and img on the 3rd dimension
        if len(imgs) > 0:
            img = np.concatenate(imgs, axis=2)
    elif type(img) is str:
        img = tifffile.imread(img)

    # convert to 0-255
    img = normalize_image(img, convert_to_uint8=True)

    height, width, _ = img.shape

    if cell_polygons_df is None:
        cell_polygons_df = adata.obsm['cell_boundaries']

    if nucleus_polygons_df is None:
        nucleus_polygons_df = adata.obsm['nucleus_boundaries']

    polygons_df = pd.merge(cell_polygons_df, nucleus_polygons_df, left_index=True, right_index=True, suffixes=('_cell', '_nucleus'))

    # Use existing Polygon objects directly
    cell_geometries = {
        cell_id: {
            'cell_polygon': row['geometry_cell'],
            'cell_bounds': row['geometry_cell'].bounds,
            'nucleus_polygon': row['geometry_nucleus'],
            'nucleus_bounds': row['geometry_nucleus'].bounds,
        }
        for cell_id, row in polygons_df.iterrows()
    }

    # Calculate sliding window positions
    step_size = patch_size - overlap
    patch_y = range(0, height - patch_size + 1, step_size)
    patch_x = range(0, width - patch_size + 1, step_size)
    patch_y = sorted(set(list(patch_y) + [height - patch_size]))
    patch_x = sorted(set(list(patch_x) + [width - patch_size]))

    # Create a list of all patches
    if selected_patches is None:
        all_patches = []
        for i, y in enumerate(patch_y):
            for j, x in enumerate(patch_x):
                patch_id = f"patch_{i}_{j}"
                all_patches.append((y, x, patch_id))
        logger.info(f"Image size: {height}x{width}, Window size: {patch_size}, Overlap: {overlap}")
        logger.info(f"Total windows: {len(all_patches)}")
        if test_mode:
            random.seed(random_seed)  # For reproducibility
            selected_patches = random.sample(all_patches, min(npatches, len(all_patches)))
            logger.info(f"Processing {len(selected_patches)} random patches")
        else:
            selected_patches = all_patches
    # logger.info(f"Processing {len(selected_patches)} patches using {n_jobs} parallel jobs")

    # Create output directories if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if extract_cell_image:
            os.makedirs(os.path.join(output_dir, "cell_images"), exist_ok=True)
        if extract_patch_image:
            os.makedirs(os.path.join(output_dir, "patch_images"), exist_ok=True)

    # Define the worker function to process a single window
    def process_window(y, x, patch_id):
        patch_bounds = box(x, y, x+patch_size, y+patch_size)
        patch_img = img[y:y+patch_size, x:x+patch_size]
        # Find cells intersecting this window using spatial predicate
        cells_in_patch = [
            (cell_id, geo['cell_polygon'], geo['nucleus_polygon'])
            for cell_id, geo in cell_geometries.items()
            if geo['cell_polygon'].intersects(patch_bounds)
        ]

        if len(cells_in_patch) == 0 and skip_empty_patch:
            logger.info(f"{patch_id} is empty, skipping...", flush=True)
            return None, {}, {}, set()

        cell_ids_in_window = set([cell_id for (cell_id, _, _) in cells_in_patch])

        # Create visualization images
        cell_images_result, patch_images_result = {}, {}

        if extract_cell_image or extract_patch_image:
            if output_dir is not None and extract_patch_image:
                patch_dir = os.path.join(output_dir, "patch_images")
                p_patch_orignal_img = os.path.join(patch_dir, f"{patch_id}_original.tif")
                p_patch_masked_img = os.path.join(patch_dir, f"{patch_id}_masked.tif")
                p_patch_boundary_img = os.path.join(patch_dir, f"{patch_id}_boundary.tif")
                if not overwrite and os.path.exists(p_patch_orignal_img) and os.path.exists(p_patch_boundary_img):
                    logger.info(f"Patch {patch_id} already exists, skipping...", flush=True)
                    return patch_id, {}, {}, cell_ids_in_window

            patch_cell_mask = np.zeros(patch_img.shape[:2], dtype=np.uint8)
            patch_nucleus_mask = np.zeros(patch_img.shape[:2], dtype=np.uint8)
            patch_image_w_boundary = patch_img[:, :, :3].copy()

            if output_dir is not None and extract_cell_image:
                cell_dir = os.path.join(output_dir, "cell_images")

            for cell_id, cell_polygon, nucleus_polygon in cells_in_patch:
                if output_dir is not None and extract_cell_image:
                    p_cell_unmasked = os.path.join(cell_dir, f"{cell_id}_original.tif")
                    p_cell_masked = os.path.join(cell_dir, f"{cell_id}_masked.tif")
                    p_cell_boundary = os.path.join(cell_dir, f"{cell_id}_boundary.tif")
                    if not overwrite and not extract_patch_image and os.path.exists(p_cell_unmasked) and os.path.exists(p_cell_masked) and os.path.exists(p_cell_boundary):
                        logger.info(f"Cell {cell_id} already exists, skipping...", flush=True)
                        continue

                # Convert global polygon to window coordinates
                local_cell_poly = translate(cell_polygon, xoff=-x, yoff=-y)
                int_cell_coords = np.array(local_cell_poly.exterior.coords.xy).T.astype(int)
                local_nucleus_poly = translate(nucleus_polygon, xoff=-x, yoff=-y)
                int_nucleus_coords = np.array(local_nucleus_poly.exterior.coords.xy).T.astype(int)
                # Create patch mask for cell and nucleus, and draw boundaries
                if int_cell_coords.shape[0] > 2:  # Need at least 3 points
                    cv2.fillPoly(patch_cell_mask, [int_cell_coords], 255)
                    cv2.polylines(patch_image_w_boundary, [int_cell_coords],
                            isClosed=True, color=cell_boundary_color, thickness=2)

                if int_nucleus_coords.shape[0] > 2:  # Need at least 3 points
                    cv2.fillPoly(patch_nucleus_mask, [int_nucleus_coords], 255)
                    cv2.polylines(patch_image_w_boundary, [int_nucleus_coords],
                            isClosed=True, color=nucleus_boundary_color, thickness=2)

                # Extract cell image
                if extract_cell_image:
                    if cell_id not in cell_images_result:
                        centroid = cell_polygon.centroid.coords[0]
                        # Calculate square bounds around centroid
                        half_size = cell_size // 2
                        y_min = int(centroid[1] - half_size)
                        y_max = int(centroid[1] + half_size)
                        x_min = int(centroid[0] - half_size)
                        x_max = int(centroid[0] + half_size)

                        cell_img, cell_mask, local_cell_poly, nucleus_mask, local_nucleus_poly = extract_and_pad_cell(
                            img, x_min, x_max, y_min, y_max,
                            cell_size, cell_polygon,
                            nucleus_polygon=nucleus_polygon
                        )

                        cell_image_tmp = create_cell_images(
                            cell_img, cell_mask, local_cell_poly,
                            cell_boundary_color=cell_boundary_color,
                            nucleus_mask=nucleus_mask,
                            local_nucleus_poly=local_nucleus_poly,
                            nucleus_boundary_color=nucleus_boundary_color,
                        )
                        cell_image_tmp['cell_id'] = cell_id
                        cell_image_tmp['patch_id'] = patch_id

                        if output_dir is not None:
                            cell_image_tmp['p_cell_unmasked'] = p_cell_unmasked
                            cell_image_tmp['p_cell_masked'] = p_cell_masked
                            cell_image_tmp['p_cell_boundary'] = p_cell_boundary
                            # Save cell images directly in the worker
                            tifffile.imwrite(p_cell_unmasked, cell_image_tmp['unmasked'])
                            if save_all_images:
                                tifffile.imwrite(p_cell_masked, cell_image_tmp['masked'])
                                tifffile.imwrite(p_cell_boundary, cell_image_tmp['boundary'])
                        cell_images_result[cell_id] = cell_image_tmp

            patch_unmasked_img = np.concatenate([
                patch_img,
                patch_cell_mask.reshape((patch_cell_mask.shape[0], patch_cell_mask.shape[1], 1)),
                patch_nucleus_mask.reshape((patch_nucleus_mask.shape[0], patch_nucleus_mask.shape[1], 1))
            ], axis=2)
            patch_masked_img = cv2.bitwise_and(patch_img, patch_img, mask=patch_cell_mask)

            # Store patch data
            patch_image_tmp = {
                'patch_id': patch_id,
                'unmasked': patch_unmasked_img,
                'masked': patch_masked_img,
                'boundary': patch_image_w_boundary,
                'cells': [cid for (cid, _, _) in cells_in_patch],
            }

            if output_dir is not None and extract_patch_image:
                patch_image_tmp['p_patch_unmasked_img'] = p_patch_orignal_img
                patch_image_tmp['p_patch_masked_img'] = p_patch_masked_img
                patch_image_tmp['p_patch_boundary_img'] = p_patch_boundary_img
                # Save patch images directly in the worker
                tifffile.imwrite(p_patch_orignal_img, patch_unmasked_img)
                if save_all_images:
                    tifffile.imwrite(p_patch_masked_img, patch_masked_img)
                    tifffile.imwrite(p_patch_boundary_img, patch_image_w_boundary)

            patch_images_result[patch_id] = patch_image_tmp
        logger.info(f"{patch_id}: {len(cells_in_patch)} cells found.", flush=True)
        logging.info(f"{patch_id}: {len(cells_in_patch)} cells found.")
        if not return_images:
            cell_images_result = {}
            patch_images_result = {}
        gc.collect()
        return patch_id, cell_images_result, patch_images_result, cell_ids_in_window

    # Batch the patches
    def batch_patches(patches, batch_size):
        for i in range(0, len(patches), batch_size):
            yield patches[i:i+batch_size]

    # Worker function for a batch of patches
    def process_patch_batch(patch_batch):
        batch_results = []
        for y, x, patch_id in patch_batch:
            result = process_window(y, x, patch_id)
            batch_results.append(result)
        return batch_results

    if batch_size * n_jobs > len(selected_patches):
        logger.info(f"Warning: batch size ({batch_size}) * n_jobs ({n_jobs}) > number of patches ({len(selected_patches)}), reducing n_jobs to {len(selected_patches) // batch_size}")
        batch_size = len(selected_patches) // n_jobs

    patch_batches = list(batch_patches(selected_patches, batch_size))
    logger.info(f"Processing {len(selected_patches)} patches in {len(patch_batches)} batches using {n_jobs} parallel jobs")

    with parallel_backend('loky', n_jobs=n_jobs):
        results_batches = Parallel(verbose=10)(
            delayed(process_patch_batch)(batch)
            for batch in tqdm(patch_batches, desc="Processing patch batches")
        )

    # Flatten the list of results
    results = [item for batch in results_batches for item in batch]

    # Combine results
    patch_cell_mapping = {}
    all_cell_images = {}
    all_patch_images = {}

    for patch_id, cell_images_result, patch_images_result, cell_ids in results:
        if patch_id is not None:
            patch_cell_mapping[patch_id] = cell_ids
            all_cell_images.update(cell_images_result)
            all_patch_images.update(patch_images_result)

    return adata, all_cell_images, all_patch_images, patch_cell_mapping


def plot_extracted_images(cell_images, patch_images, output_dir=None):
    logger.info(list(patch_images.keys()))
    # logger.info(list(patch_images.values())[0])
    logger.info(list(patch_images.values())[0]['unmasked'].shape)
    n = 2 + list(patch_images.values())[0]['unmasked'].shape[-1] - 3
    fig, axs = plt.subplots(n, len(patch_images), figsize=(5*len(patch_images), 5*n))
        
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]

    if len(axs.shape)!=2:
        axs = np.array([axs]).T
        
    for i,(patch_id, v) in enumerate(patch_images.items()):
        ax = axs[0, i]
        ax.imshow(v['unmasked'][:,:,:3])
        ax.axis("off")
        
        ax = axs[1, i]
        ax.imshow(v['boundary'][:,:,:3])
        ax.axis("off")
        
        ax = axs[2, i]
        ax.imshow(v['masked'][:,:,:3])
        ax.axis("off")
        
        for j,k in enumerate(range(3, v['unmasked'].shape[-1])):
            ax = axs[j+2, i]
            ax.imshow(v['unmasked'][:,:,k], cmap='grey')
            ax.axis("off")
        
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/patch_images.png", dpi=300)
    plt.show()

    n = 3 + list(cell_images.values())[0]['unmasked'].shape[-1] - 3
    fig, axs = plt.subplots(n,10,figsize=(5*10, 5*n))

    for i, (cell_id, v) in enumerate(list(cell_images.items())[:10]):
        axs[0, i].imshow(v['unmasked'][:,:,:3])
        axs[0, i].axis("off")
        axs[1, i].imshow(v['boundary'][:,:,:3])
        axs[1, i].axis("off")
        axs[2, i].imshow(v['masked'][:,:,:3])
        axs[2, i].axis("off")
        for j,k in enumerate(range(3, v['unmasked'].shape[-1])):
            ax = axs[j+3, i]
            ax.imshow(v['unmasked'][:,:,k], cmap='grey')
            ax.axis("off")

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/cell_images.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Prepare Xenium data")
    parser.add_argument('--input_path', type=str, required=False, help="Path to the input data")
    parser.add_argument('--adata', type=str, required=False, help="Path to the AnnData object")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--extra_img_keys', type=str, nargs='*', default=None, help="Additional image keys to load")
    parser.add_argument('--test_mode', action='store_true', help="Run in test mode")
    parser.add_argument('--npatches', type=int, default=10, help="Number of patches to test")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--cell_size', type=int, default=256, help="Size of the cell square")
    parser.add_argument('--patch_size', type=int, default=256, help="Size of the sliding window")
    parser.add_argument('--overlap', type=int, default=100, help="Overlap between windows")
    parser.add_argument('--skip_empty_patch', action='store_true', help="Skip empty patches")
    parser.add_argument('--extract_cell_image', action='store_true', help="Extract cell images")
    parser.add_argument('--extract_patch_image', action='store_true', help="Extract patch images")
    parser.add_argument('--return_images', action='store_true', help="Return images")
    parser.add_argument('--save_all_images', action='store_true', help="Save all images")
    parser.add_argument('--cell_boundary_color', type=tuple, default=(0, 255, 0), help="Color for cell boundaries")
    parser.add_argument('--nucleus_boundary_color', type=tuple, default=(255, 0, 0), help="Color for nucleus boundaries")
    parser.add_argument('--write_gene_expression', action='store_true', help="Write gene expression data")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs to run")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    if args.adata is not None:
        if args.adata.endswith("h5ad"):
            adata = sc.read(args.adata)
        elif args.adata.endswith("pickle"):
            with open(args.adata, 'rb') as f:
                adata = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {args.adata}")
    else:
        adata = load_xenium_data(
            input_path=args.input_path,
        )

    if args.write_gene_expression:
        adata.to_df.to_csv(f"{args.output_dir}/normalized.csv")

    adata, cell_images, patch_images, patch_cell_mapping = load_xenium_data_with_polygons_parallel(
        adata,
        extra_img_keys=args.extra_img_keys,
        patch_size=args.patch_size,
        cell_size=args.cell_size,
        overlap=args.overlap,
        test_mode=args.test_mode,
        npatches=args.npatches,
        extract_cell_image=args.extract_cell_image,
        extract_patch_image=args.extract_patch_image,
        output_dir=f"{args.output_dir}/input",
        overwrite=args.overwrite,
        save_all_images=args.save_all_images,
        return_images=args.return_images,
        skip_empty_patch=args.skip_empty_patch,
        verbose=args.verbose,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        cell_boundary_color=args.cell_boundary_color,
        nucleus_boundary_color=args.nucleus_boundary_color,
    )

    if cell_images is not None:
        cell_image_paths = {}
        for cell_id, v in cell_images.items():
            cell_image_paths[cell_id] = v['p_cell_unmasked']
        with open(f"{args.output_dir}/input/cell_image_paths.json", 'w') as f:
            json.dump(cell_image_paths, f)
    
    if patch_images is not None:
        patch_image_paths = {}
        for patch_id, v in patch_images.items():
            patch_image_paths[patch_id] = v['p_patch_unmasked_img']
        with open(f"{args.output_dir}/input/patch_image_paths.json", 'w') as f:
            json.dump(patch_image_paths, f)
        
    if patch_cell_mapping is not None:
        patch_cell_mapping = {k: list(v) for k,v in patch_cell_mapping.items()}
        with open(f"{args.output_dir}/input/patch_cell_mapping.json", 'w') as f:
            json.dump(patch_cell_mapping, f)
    
    if cell_images is not None and len(cell_images)>0 and \
        patch_images is not None and len(patch_images)>0:
        plot_extracted_images(cell_images, patch_images, output_dir=f"{args.output_dir}/input")

    return

if __name__ == '__main__':
    main()
