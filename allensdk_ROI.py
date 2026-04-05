import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from utils.atlas_utils import (
    ensure_atlas_data,
    project,
    create_colored_mask_from_targets,
    load_image,
    save_results
)

def process_single_slice(row: pd.Series, annot: np.ndarray, id_mapping: dict, 
                          target_ids_int: list, image_dir: str, roi_dir: str, show_plot: bool):
    """
    Process a single brain slice image.
    
    This function handles the full pipeline for one image: file lookup, image loading,
    coordinate mapping, ROI extraction, optional visualization, and saving results.

    Parameters
    ----------
    row : pd.Series
        A row from the DeepSlice CSV containing alignment coordinates.
    annot : np.ndarray
        3D Allen Brain Atlas annotation volume.
    id_mapping : dict
        Mapping from atlas ID to target ID (for merging sub-regions).
    target_ids_int : list of int
        List of target region IDs specified by the user.
    image_dir : str
        Path to the directory containing input images.
    roi_dir : str
        Path to the directory for saving output results.
    show_plot : bool
        Whether to display a Matplotlib window for visualization.
    """
    filename = row["Filenames"]
    name_stem = '.'.join(filename.split('.')[:-1])
    print(f"\n===== Processing: {filename} =====")

    img_path = None
    img_ext = None
    for ext in ['.tif', '.png']:
        candidate = os.path.join(image_dir, f"{name_stem}{ext}")
        if os.path.exists(candidate):
            img_path, img_ext = candidate, ext
            break
    
    if img_path is None:
        print(f"  Skipping: no .tif or .png file found")
        return

    img_2d = load_image(img_path, img_ext)
    w, h = int(row["width"]), int(row["height"])
    
    O_csv = np.array([row["ox"], row["oy"], row["oz"]])
    U_csv = np.array([row["ux"], row["uy"], row["uz"]])
    V_csv = np.array([row["vx"], row["vy"], row["vz"]])

    atlas_slice = project(annot, O_csv, U_csv, V_csv, w, h)
    
    target_labeled_slice = np.zeros_like(atlas_slice, dtype=np.int32)
    for atlas_id, target_id in id_mapping.items():
        target_labeled_slice[atlas_slice == atlas_id] = target_id

    present_ids = set(np.unique(target_labeled_slice))
    present_ids.discard(0)
    
    pixel_counts = {tid: 0 for tid in target_ids_int}
    for tid in target_ids_int:
        count = int(np.sum(target_labeled_slice == tid))
        pixel_counts[tid] = count
        if count == 0:
            print(f"Region ID {tid} is not present in the image.")

    rgb_mask, color_mapping = create_colored_mask_from_targets(target_labeled_slice, target_ids_int)

    if show_plot:
        f, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(img_2d, cmap='gray'); axes[0].set_title("Original"); axes[0].axis('off')
        axes[1].imshow(atlas_slice, cmap='tab20', vmin=0, vmax=2000); axes[1].set_title("Allen Projection"); axes[1].axis('off')
        axes[2].imshow(rgb_mask); axes[2].set_title("Final ROI Mask"); axes[2].axis('off')
        plt.tight_layout(); plt.show()

    if len(present_ids) > 0:
        save_results(rgb_mask, color_mapping, pixel_counts, target_ids_int, roi_dir, name_stem)
    else:
        print("No ROI found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='./raw_data',
                        help='Path to image directory (default: ./raw_data)')
    parser.add_argument('--region-ids', type=str, nargs='+', required=True,
                        help='List of Allen Atlas region IDs (e.g., --region-ids 382 463 525)')
    parser.add_argument('--show-plot', action='store_true', default=False,
                        help='Show visualization plot (default: False)')
    args = parser.parse_args()
    image_dir = args.image_dir
    region_ids = args.region_ids
    show_plot = args.show_plot

    roi_dir = os.path.join(image_dir, "ROI_data")
    os.makedirs(roi_dir, exist_ok=True)
    
    root = os.path.dirname(os.path.abspath(__file__)) 
    manifest_path = Path(root) / 'manifest.json'
    RESOLUTION = 25

    print(">>> Initializing Allen Brain Atlas...")
    mcc = ensure_atlas_data(str(manifest_path), RESOLUTION)
    if mcc is None:
        print("Cannot proceed, missing atlas data")
        return

    annot, _ = mcc.get_annotation_volume()
    structure_tree = mcc.get_structure_tree()
    
    id_mapping = {}
    target_ids_int = [int(rid) for rid in region_ids]
    
    print(f">>> Processing {len(region_ids)} brain regions...")
    for rid_str in region_ids:
        rid = int(rid_str)
        try:
            ids = structure_tree.descendant_ids([rid])[0]
            for did in ids:
                id_mapping[did] = rid
            print(f"  - Region ID {rid}: {len(ids)} descendant regions mapped to it")
        except Exception as e:
            print(f"  [Warning] Failed to get descendants for region {rid}: {e}")

    csv_path = os.path.join(image_dir, "MyResults.csv")
    if not os.path.exists(csv_path):
        print(f"Result file not found: {csv_path}")
        print("Please ensure DeepSlice step has been run successfully.")
        return
    pred = pd.read_csv(csv_path)

    print(f">>> Starting batch processing for {len(pred)} images...")
    for _, row in pred.iterrows():
        process_single_slice(row, annot, id_mapping, target_ids_int, image_dir, roi_dir, show_plot)
    


if __name__ == "__main__":
    main()