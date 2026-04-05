import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
from typing import Tuple, Optional, List
import argparse

def get_clahe_image(
    img_channel: np.ndarray, 
    clip_limit: float = 0.8, 
    grid_size: int = 16, 
    trim_percent: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes CLAHE enhanced image.
    
    Args:
        img_channel: Input grayscale image channel.
        clip_limit: CLAHE contrast limit.
        grid_size: CLAHE grid size.
        trim_percent: Percentile for outlier trimming.
        
    Returns:
        tuple: (visualization_gray, enhanced_target)
            - visualization_gray: uint8 image for display
            - enhanced_target: Image in original dtype/range for segmentation
    """
    lower = np.percentile(img_channel, trim_percent)
    upper = np.percentile(img_channel, 100 - trim_percent)
    img_clipped = np.clip(img_channel, lower, upper)
    
    img_norm = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    visualization_gray = clahe.apply(img_norm)
    
    enhanced_target = cv2.normalize(visualization_gray, None, lower, upper, cv2.NORM_MINMAX)
    enhanced_target = enhanced_target.astype(img_channel.dtype)
    
    return visualization_gray, enhanced_target

def split_image_into_tiles(
    img: np.ndarray, 
    max_longest_edge: int = 8000
) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """Split large image into manageable tiles.
    
    Args:
        img: Input image.
        max_longest_edge: Maximum allowed edge length for a tile.
        
    Returns:
        list: List of tuples (tile, y_offset, x_offset, tile_height, tile_width).
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_longest_edge:
        return [(img, 0, 0, h, w)]
    n = 1
    while max(h, w) / (2 ** n) > max_longest_edge:
        n += 1
    s = 2 ** n
    tiles = []
    dh, dw = h // s, w // s
    for i in range(s):
        for j in range(s):
            y0, y1 = i * dh, min((i + 1) * dh, h)
            x0, x1 = j * dw, min((j + 1) * dw, w)
            tile = img[y0:y1, x0:x1]
            tiles.append((tile, y0, x0, y1 - y0, x1 - x0))
    return tiles

def show_contrast_comparison(
    original_img: np.ndarray, 
    clahe_img: np.ndarray, 
    name_stem: str,
    trim_percent: float = 1.0
) -> None:
    """Show a side-by-side comparison plot of original vs CLAHE enhanced image.
    
    Args:
        original_img: Original grayscale image channel.
        clahe_img: CLAHE enhanced uint8 image.
        name_stem: Image filename stem for plot title.
        trim_percent: Percentile for outlier trimming when displaying original image.
    """
    lower = np.percentile(original_img, trim_percent)
    upper = np.percentile(original_img, 100 - trim_percent)
    original_display = cv2.normalize(np.clip(original_img, lower, upper), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_display, cmap='gray')
    plt.title(f"Original Image: {name_stem}", fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(clahe_img, cmap='gray')
    plt.title(f"CLAHE Enhanced: {name_stem}", fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(
    tif_path: str, 
    args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load image and generate processed channels for segmentation and visualization.
    
    Args:
        tif_path: Path to input TIFF image.
        args: Parsed command line arguments.
        
    Returns:
        tuple: (enhanced_target, visualization_gray, H, W)
    """
    img = np.asarray(tifffile.imread(tif_path))
    if img.ndim == 3:
        img = np.transpose(img, (1,2,0))
        target_channel = img[:, :, 1] if img.shape[2] >= 2 else img[:, :, 0]
    else:
        target_channel = img
    H, W = target_channel.shape[:2]

    original_target = target_channel.copy()
    
    if args.enhance_contrast:
        visualization_gray, enhanced_target = get_clahe_image(
            target_channel,
            clip_limit=args.clahe_clip,
            grid_size=args.clahe_grid,
            trim_percent=args.trim_percent
        )
    else:
        enhanced_target = target_channel
        lower = np.percentile(original_target, args.trim_percent)
        upper = np.percentile(original_target, 100 - args.trim_percent)
        visualization_gray = cv2.normalize(np.clip(original_target, lower, upper), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if args.show_plot:
        name_stem = os.path.basename(tif_path)
        name_stem = '.'.join(name_stem.split('.')[:-1])
        show_contrast_comparison(original_target, visualization_gray, name_stem, args.trim_percent)

    return enhanced_target, visualization_gray, H, W