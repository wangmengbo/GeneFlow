# rectified/src/stain_normalization.py
import numpy as np
from skimage import exposure
import logging

logger = logging.getLogger(__name__)

def normalize_staining_rgb_skimage_hist_match(source_rgb_img_np, target_rgb_img_np):
    """
    Normalizes the staining of the source RGB image (generated) to match
    the target RGB image (real) using per-channel histogram matching
    from scikit-image. Operates only on the 3 RGB channels.

    Args:
        source_rgb_img_np (np.ndarray): The source RGB image (H, W, 3) as a NumPy array.
                                        Expected float [0,1] or uint8 [0,255].
        target_rgb_img_np (np.ndarray): The target RGB image (H, W, 3) for reference,
                                        as a NumPy array. Expected float [0,1] or uint8 [0,255].

    Returns:
        np.ndarray: The histogram-matched source RGB image (H, W, 3) in the same
                    data type and range as the input source_rgb_img_np.
                    Returns the original source_rgb_img_np if normalization fails.
    """
    if source_rgb_img_np.ndim != 3 or source_rgb_img_np.shape[2] != 3 or \
       target_rgb_img_np.ndim != 3 or target_rgb_img_np.shape[2] != 3:
        logger.error(f"Input images for histogram matching must be (H, W, 3). Got source: {source_rgb_img_np.shape}, target: {target_rgb_img_np.shape}")
        return source_rgb_img_np

    original_dtype = source_rgb_img_np.dtype
    # Determine if input was float [0,1] to restore range correctly
    source_was_float_01 = np.issubdtype(original_dtype, np.floating) and \
                          (source_rgb_img_np.min() >= -1e-3 and source_rgb_img_np.max() <= 1.0 + 1e-3)

    matched_img_np = np.zeros_like(source_rgb_img_np)

    try:
        # Ensure inputs for match_histograms are float [0,1] for consistent behavior
        source_norm = source_rgb_img_np.astype(np.float32)
        target_norm = target_rgb_img_np.astype(np.float32)

        if not source_was_float_01 and source_rgb_img_np.max() > 1.0: # Assuming uint8 scaled to float
            source_norm /= 255.0
        source_norm = np.clip(source_norm, 0, 1)

        target_was_float_01 = np.issubdtype(target_rgb_img_np.dtype, np.floating) and \
                             (target_rgb_img_np.min() >= -1e-3 and target_rgb_img_np.max() <= 1.0 + 1e-3)
        if not target_was_float_01 and target_rgb_img_np.max() > 1.0:
            target_norm /= 255.0
        target_norm = np.clip(target_norm, 0, 1)

        for channel_idx in range(3): # Iterate through R, G, B
            source_channel = source_norm[:, :, channel_idx]
            target_channel = target_norm[:, :, channel_idx]

            matched_channel = exposure.match_histograms(source_channel, target_channel)
            matched_img_np[:, :, channel_idx] = matched_channel

        # Restore original data type and range
        if source_was_float_01:
            matched_img_np = np.clip(matched_img_np, 0, 1).astype(original_dtype)
        elif np.issubdtype(original_dtype, np.unsignedinteger): # e.g. uint8
            matched_img_np = np.clip(matched_img_np * 255.0, 0, 255).astype(original_dtype)
        else: # Other float types or unhandled integer types
            matched_img_np = matched_img_np.astype(original_dtype) # trust clip for float, or user to ensure uint is handled

        logger.debug("Per-channel histogram matching applied successfully.")
        return matched_img_np

    except Exception as e:
        logger.error(f"Error during scikit-image histogram matching: {e}", exc_info=True)
        logger.warning("Returning original source image due to histogram matching error.")
        return source_rgb_img_np