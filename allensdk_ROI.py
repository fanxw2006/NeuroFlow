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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='./raw_data',
                        help='Path to image directory (default: ./raw_data)')
    parser.add_argument('--region-ids', type=str, nargs='+', required=True,
                        help='List of Allen Atlas region IDs (e.g., --region-ids 382 463 525)')
    parser.add_argument('--show-plot', action='store_true', default=False,
                        help='Show visualization plot (default: False)')
    return parser.parse_args()


def setup_atlas_mapping(region_ids: list[str], manifest_path: Path, resolution: int = 25):
    """
    Step 1: Load Atlas data and setup ID mapping (Descendants -> Target ID).
    Extracted from main to reduce complexity.
    """
    mcc = ensure_atlas_data(str(manifest_path), resolution)
    if mcc is None:
        return None, None, None

    print("Loading Allen Atlas (from cache)")
    annot, _ = mcc.get_annotation_volume()
    structure_tree = mcc.get_structure_tree()
    
    print(f"Processing {len(region_ids)} brain regions in batch mode")
    
    id_mapping = {}
    target_ids_int = [int(rid) for rid in region_ids]
    
    for rid_str in region_ids:
        rid = int(rid_str)
        try:
            ids = structure_tree.descendant_ids([rid])[0]
            for did in ids:
                id_mapping[did] = rid
            print(f"  - Region ID {rid}: {len(ids)} descendant regions mapped to it")
        except Exception as e:
            print(f"  [Warning] Failed to get descendants for region {rid}: {e}")
            
    return annot, target_ids_int, id_mapping


def process_single_slice(row: pd.Series, annot: np.ndarray, id_mapping: dict, 
                          target_ids_int: list, image_dir: str, roi_dir: str, show_plot: bool):
    """
    Step 2: Process a single image row from the CSV.
    Extracted from main loop.
    """
    filename = row["Filenames"]
    name_stem = '.'.join(filename.split('.')[:-1])
    print(f"\n===== Processing: {filename} =====")

    # Find image file (Simplified logic)
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

    # Load data
    img_2d = load_image(img_path, img_ext)
    w, h = int(row["width"]), int(row["height"])
    
    O_csv = np.array([row["ox"], row["oy"], row["oz"]])
    U_csv = np.array([row["ux"], row["uy"], row["uz"]])
    V_csv = np.array([row["vx"], row["vy"], row["vz"]])

    # Projection and Labeling
    atlas_slice = project(annot, O_csv, U_csv, V_csv, w, h)
    
    target_labeled_slice = np.zeros_like(atlas_slice, dtype=np.int32)
    for atlas_id, target_id in id_mapping.items():
        target_labeled_slice[atlas_slice == atlas_id] = target_id

    # Analysis
    present_ids = set(np.unique(target_labeled_slice))
    present_ids.discard(0)
    
    # Check missing & count pixels (Combined loop logic)
    pixel_counts = {tid: 0 for tid in target_ids_int}
    for tid in target_ids_int:
        count = int(np.sum(target_labeled_slice == tid))
        pixel_counts[tid] = count
        if count == 0:
            print(f"Region ID {tid} is not present in the image.")

    # Visualization
    rgb_mask, color_mapping = create_colored_mask_from_targets(target_labeled_slice, target_ids_int)

    if show_plot:
        f, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(img_2d, cmap='gray'); axes[0].set_title("Original"); axes[0].axis('off')
        axes[1].imshow(atlas_slice, cmap='tab20', vmin=0, vmax=2000); axes[1].set_title("Allen Projection"); axes[1].axis('off')
        axes[2].imshow(rgb_mask); axes[2].set_title("Final ROI Mask"); axes[2].axis('off')
        plt.tight_layout(); plt.show()

    # Saving
    if len(present_ids) > 0:
        save_results(rgb_mask, color_mapping, pixel_counts, target_ids_int, roi_dir, name_stem)
    else:
        print("No ROI found.")


def run_atlas(region_ids: list[str], image_dir: str, show_plot: bool):
    """
    Main workflow - Now highly simplified, acts as the conductor.
    """
    # Setup Paths
    roi_dir = os.path.join(image_dir, "ROI_data")
    os.makedirs(roi_dir, exist_ok=True)
    
    root = os.path.dirname(os.path.abspath(__file__)) 
    manifest_path = Path(root) / 'manifest.json'

    # Phase 1: Atlas Setup
    annot, target_ids_int, id_mapping = setup_atlas_mapping(region_ids, manifest_path)
    if annot is None:
        print("Cannot proceed, missing atlas data")
        return

    # Phase 2: Load CSV
    csv_path = os.path.join(image_dir, "MyResults.csv")
    if not os.path.exists(csv_path):
        print(f"Result file not found: {csv_path}")
        print("Please ensure DeepSlice step has been run successfully.")
        return
    pred = pd.read_csv(csv_path)

    # Phase 3: Process Images
    for _, row in pred.iterrows():
        process_single_slice(row, annot, id_mapping, target_ids_int, image_dir, roi_dir, show_plot)


if __name__ == "__main__":
    args = parse_args()
    run_atlas(args.region_ids, args.image_dir, args.show_plot)