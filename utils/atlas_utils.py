import os
import csv
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy.ndimage import map_coordinates
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def ensure_atlas_data(manifest_path: str, resolution: int = 25, log: callable = print) -> MouseConnectivityCache:
    """
    Ensure Allen Brain Atlas data is downloaded and cached.
    
    Args:
        manifest_path: Path to manifest file for atlas cache
        resolution: Atlas resolution in microns
        log: Logging function
    
    Returns:
        MouseConnectivityCache object if successful, None otherwise
    """
    log("===== Checking Allen Brain Atlas Data =====")
    
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir and not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir, exist_ok=True)

    try:
        mcc = MouseConnectivityCache(
            resolution=resolution, 
            manifest_file=manifest_path
        )
        
        log("Loading/downloading atlas data (this may take a few minutes, please wait)...")
        annot, _ = mcc.get_annotation_volume()
        
        log(f"[OK] Atlas data loaded, shape: {annot.shape}")
        return mcc
        
    except Exception as e:
        log(f"[Error] Atlas data preparation failed: {e}")
        log("[Tip] If download was interrupted, check network and rerun, program will resume.")
        return None


def remap_and_align(O_csv: np.ndarray, U_csv: np.ndarray, V_csv: np.ndarray, 
                    vol_shape: tuple, flip_z: bool = True, flip_y: bool = True, flip_x: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remap and align coordinate system to match volume dimensions.
    
    Args:
        O_csv: Origin coordinates from CSV
        U_csv: First basis vector from CSV
        V_csv: Second basis vector from CSV
        vol_shape: Shape of target volume (Z,Y,X)
        flip_z: Whether to flip Z axis (default: True)
        flip_y: Whether to flip Y axis (default: True)
        flip_x: Whether to flip X axis (default: True)
    
    Returns:
        Tuple of (new_O, new_U, new_V) after remapping and alignment
    """
    csv_x, csv_y, csv_z = O_csv
    csv_ux, csv_uy, csv_uz = U_csv
    csv_vx, csv_vy, csv_vz = V_csv

    new_O = np.array([csv_y, csv_z, csv_x])
    new_U = np.array([csv_uy, csv_uz, csv_ux])
    new_V = np.array([csv_vy, csv_vz, csv_vx])

    if flip_z:
        new_O[0] = vol_shape[0] - new_O[0]
        new_U[0] = -new_U[0]
        new_V[0] = -new_V[0]
    if flip_y:
        new_O[1] = vol_shape[1] - new_O[1]
        new_U[1] = -new_U[1]
        new_V[1] = -new_V[1]
    if flip_x:
        new_O[2] = vol_shape[2] - new_O[2]
        new_U[2] = -new_U[2]
        new_V[2] = -new_V[2]

    return new_O, new_U, new_V


def project(volume: np.ndarray, O_csv: np.ndarray, U_csv: np.ndarray, V_csv: np.ndarray, 
            width: int, height: int) -> np.ndarray:
    """
    Project 3D volume onto 2D plane using defined coordinate system.
    
    Args:
        volume: 3D input volume
        O_csv: Origin coordinates
        U_csv: First basis vector
        V_csv: Second basis vector
        width: Output image width
        height: Output image height
    
    Returns:
        2D projected image
    """
    O, U, V = remap_and_align(O_csv, U_csv, V_csv, volume.shape)
    px = np.arange(width)
    py = np.arange(height)
    grid_px, grid_py = np.meshgrid(px, py, indexing='xy')
    px_norm = grid_px / width
    py_norm = grid_py / height

    Pz = O[0] + px_norm * U[0] + py_norm * V[0]
    Py = O[1] + px_norm * U[1] + py_norm * V[1]
    Px = O[2] + px_norm * U[2] + py_norm * V[2]

    coords = np.vstack([Pz.ravel(), Py.ravel(), Px.ravel()])
    for i in range(3):
        coords[i] = np.clip(coords[i], 0, volume.shape[i] - 1)

    if np.any(volume > 0):
        print(f"Sampling range Z:{coords[0].min():.0f}~{coords[0].max():.0f}, Y:{coords[1].min():.0f}~{coords[1].max():.0f}, X:{coords[2].min():.0f}~{coords[2].max():.0f}")

    sampled = map_coordinates(volume, coords, order=0, mode="nearest")
    return sampled.reshape(height, width)


def create_colored_mask_from_targets(label_image: np.ndarray, target_ids: list) -> tuple[np.ndarray, dict]:
    """
    Converts a 2D label image (containing target IDs) to an RGB image using a distinct color per target ID.
    
    Args:
        label_image: 2D numpy array where values are target IDs.
        target_ids: List of target IDs to generate colors for.
    
    Returns:
        Tuple of (rgb_image, color_mapping_dict)
    """
    cmap = matplotlib.colormaps.get_cmap('tab20')
    
    color_mapping = {}
    n_ids = len(target_ids)
    for i, tid in enumerate(target_ids):
        color = cmap(i / n_ids)[:3] 
        color_mapping[tid] = (np.array(color) * 255).astype(np.uint8)
    
    h, w = label_image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for tid, color in color_mapping.items():
        mask = label_image == tid
        rgb_image[mask] = color
        
    return rgb_image, color_mapping


def load_image(img_path: str, img_ext: str) -> np.ndarray:
    """
    Load and standardize a 2D brain slice image.

    Parameters
    ----------
    img_path : str
    img_ext : str
    Supported extensions: '.tif' (uses tifffile) or '.png' (uses matplotlib).

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (H, W) with dtype ``np.uint8``.
    """
    if img_ext == '.tif':
        img = tifffile.imread(img_path)
    else:
        img = plt.imread(img_path)
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).astype(np.uint8)
    
    # Extract 2D slice
    if img.ndim == 3: 
        return img[0]
    return img


def save_results(rgb_mask: np.ndarray, color_mapping: dict, pixel_counts: dict, target_ids_int: list, save_dir: str, name_stem: str):
    """
    Save visualization and quantification results to disk:
    1. A PNG image of the colored ROI mask.
    2. A CSV file containing the color mapping (RGB values) and the number of 
       pixels counted for each target region ID.

    Parameters
    ----------
    rgb_mask : np.ndarray
        The colored ROI image. Shape (H, W, 3), dtype np.uint8.
    color_mapping : dict
        A dictionary mapping region IDs (int) to their RGB color tuples.
        Format: ``{region_id: (R, G, B)}`` where values are 0-255 uint8.
    pixel_counts : dict
        A dictionary mapping region IDs (int) to the number of pixels (int) 
        that were labeled as that region in the slice.
        Format: ``{region_id: count}``.
    target_ids_int : list of int
        A list of the target region IDs. This is used to enforce a specific 
        order in the output CSV rows.
    save_dir : str
        Path to the directory where files should be saved (e.g., './output').
    name_stem : str
        The base name for the output files (without extension).
        Outputs will be: ``{name_stem}_ROI_mask.png`` and ``{name_stem}_color_mapping.csv``.
    """
    png_save_path = os.path.join(save_dir, f"{name_stem}_ROI_mask.png")
    plt.imsave(png_save_path, rgb_mask)
    print(f"Color Mask saved: {png_save_path}")

    csv_mapping_path = os.path.join(save_dir, f"{name_stem}_color_mapping.csv")
    with open(csv_mapping_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['target_id', 'R', 'G', 'B', 'pixel_count'])
        for tid in target_ids_int:
            r, g, b = color_mapping[tid]
            writer.writerow([tid, r, g, b, pixel_counts[tid]])
    print(f"Color mapping table saved: {csv_mapping_path}")