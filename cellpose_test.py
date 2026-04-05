import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
from cellpose import models
from skimage.measure import label
from typing import Tuple, List, Dict, Any, Optional
from utils.image_processing import load_and_preprocess_image, split_image_into_tiles
from utils.analysis_utils import load_all_rois, analyze_and_count_cells

def run_cellpose_on_full_image(
    model: models.Cellpose,
    img_channel: np.ndarray,
    args: argparse.Namespace
) -> np.ndarray:
    """Run Cellpose prediction on full image with tiling.
    
    Args:
        model: Loaded Cellpose model.
        img_channel: Input image channel for segmentation.
        args: Parsed command line arguments.
        
    Returns:
        np.ndarray: Full instance segmentation mask.
    """
    tiles = split_image_into_tiles(img_channel, max_longest_edge=args.max_tile_edge)
    h, w = img_channel.shape[:2]
    full_predicted_mask = np.zeros((h, w), dtype=np.int32)
    current_max_label = 0

    eval_diameter = None if args.auto_diameter else 15

    print(f"Starting full-image Cellpose prediction (Image size: {h}x{w}, Tiles: {len(tiles)})...")
    start_cellpose = time.time()

    for idx, (tile_img, y_off, x_off, tile_h, tile_w) in enumerate(tiles):
        masks, flows, styles, diams = model.eval(
            tile_img,
            diameter=eval_diameter,
            channels=[0, 0],
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold
        )

        tile_max_label = masks.max()
        if tile_max_label > 0:
            tile_mask_shifted = np.where(masks > 0, masks + current_max_label, 0)
            full_predicted_mask[y_off:y_off+tile_h, x_off:x_off+tile_w] = np.maximum(
                full_predicted_mask[y_off:y_off+tile_h, x_off:x_off+tile_w],
                tile_mask_shifted
            )
            current_max_label += tile_max_label

    full_predicted_mask = label(full_predicted_mask > 0)
    print(f"Full-image Cellpose completed in {time.time() - start_cellpose:.2f}s, Total cells: {len(np.unique(full_predicted_mask)) - 1}")
    return full_predicted_mask

def save_visualizations(
    visualization_gray: np.ndarray, 
    full_cell_mask: np.ndarray, 
    roi_masks: Dict[int, np.ndarray], 
    merged_mask: np.ndarray, 
    result_dict: Dict[str, Any],
    name_stem: str,
    IMAGE_DIR: str,
    args: argparse.Namespace
) -> None:
    """Save all visualization images (merged view and individual ROI views).
    
    Args:
        visualization_gray: Base grayscale image for visualization.
        full_cell_mask: Full segmentation mask.
        roi_masks: Dictionary of ROI masks.
        merged_mask: Merged ROI mask.
        result_dict: Dictionary with cell count results.
        name_stem: Image filename stem.
        IMAGE_DIR: Root image directory.
        args: Parsed command line arguments.
    """
    mask_save_dir = os.path.join(IMAGE_DIR, "predicted_mask")
    os.makedirs(mask_save_dir, exist_ok=True)
    
    pred_binary = (full_cell_mask > 0).astype(np.uint8)
    cell_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_display_merge = cv2.cvtColor(visualization_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_display_merge, cell_contours, -1, (0, 255, 0), 1)
    
    roi_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_display_merge, roi_contours, -1, (0, 0, 255), 2)
    
    merged_cell_count = result_dict.get("ROI_总细胞数", 0)
    cv2.putText(img_display_merge, f"Merge Cells: {merged_cell_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    save_path_merge = os.path.join(mask_save_dir, f"{name_stem}_MERGED.jpg")
    cv2.imwrite(save_path_merge, img_display_merge)

    if args.show_plot:
        img_rgb = cv2.cvtColor(img_display_merge, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"Segmentation Result: {name_stem}")
        plt.axis('off')
        plt.show()

    for rid, roi_mask_binary in roi_masks.items():
        img_display_single = cv2.cvtColor(visualization_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_display_single, cell_contours, -1, (0, 255, 0), 1)
        
        single_contours, _ = cv2.findContours(roi_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_display_single, single_contours, -1, (0, 0, 255), 2)
        
        cell_count = result_dict.get(f"ROI_{rid}_细胞数", 0)
        cv2.putText(img_display_single, f"ROI: {rid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img_display_single, f"Cells: {cell_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        save_path_single = os.path.join(mask_save_dir, f"{name_stem}_ROI_{rid}.jpg")
        cv2.imwrite(save_path_single, img_display_single)

    print(f"Visualizations saved successfully")

def save_final_results(
    all_results_list: List[Dict[str, Any]], 
    IMAGE_DIR: str
) -> None:
    """Save final results to CSV.
    
    Args:
        all_results_list: List of result dictionaries for each image.
        IMAGE_DIR: Root image directory for output.
    """
    output_csv = os.path.join(IMAGE_DIR, "细胞分析结果.csv")
    df = pd.DataFrame(all_results_list)
    
    base_cols = ["图像名称", "图像高度(H)", "图像宽度(W)", "全图总细胞数", "使用模型", "分析耗时(s)"]
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    roi_cols.sort()
    merge_cols = ["ROI_总细胞数", "ROI_总面积"]
    
    new_order = base_cols + roi_cols + merge_cols
    new_order = [c for c in new_order if c in df.columns and c not in merge_cols]
    df = df[new_order]
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nAll done! Results saved to: {output_csv}")

def main() -> None:
    # Step 1: Initialize pipeline
    parser = argparse.ArgumentParser(description='Cellpose Cell Analysis Pipeline')

    parser.add_argument('--image-dir', type=str, required=True, help='Root directory for images')
    parser.add_argument('--region-ids', type=int, nargs='+', required=True, help='List of ROI IDs')
    parser.add_argument('--model-type', type=str, default='cyto3', choices=['cyto3', 'cyto2', 'nuclei'], help='Cellpose model type')
    parser.add_argument('--auto-diameter', action='store_true', help='Use automatic diameter estimation')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0, help='Cell probability threshold')
    parser.add_argument('--flow-threshold', type=float, default=0.4, help='Flow field threshold')
    parser.add_argument('--max-tile-edge', type=int, default=8000, help='Max edge length for image tiling')
    parser.add_argument('--enhance-contrast', action='store_true', default=True, help='Enable contrast enhancement')
    parser.add_argument('--clahe-clip', type=float, default=0.8, help='CLAHE contrast limit')
    parser.add_argument('--clahe-grid', type=int, default=16, help='CLAHE grid size')
    parser.add_argument('--trim-percent', type=float, default=1.0, help='Percentile for outlier trimming')
    parser.add_argument('--show-plot', action='store_true', default=False, help='Show comparison plot (Original vs CLAHE) and final segmentation result')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Computation device')
    parser.add_argument('--use-multithreading', action='store_true', help='Enable multithreading acceleration')

    args = parser.parse_args()

    threads = os.cpu_count() if args.use_multithreading else 1
    os.environ["OMP_NUM_THREADS"] = str(threads)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if args.device != "cpu":
            print(f"Warning: Specified device '{args.device}' not available, falling back to CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device.type.upper()} | Model: {args.model_type}")

    # Step 2: Set up paths and load metadata
    IMAGE_DIR = args.image_dir
    ROI_DATA_DIR = os.path.join(IMAGE_DIR, "ROI_data")
    region_ids = [int(rid) for rid in args.region_ids]

    print(f"===== Target brain region ID list: {region_ids} =====")
    all_results_list = []

    csv_path = os.path.join(IMAGE_DIR, "MyResults.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find DeepSlice results: {csv_path}")
        return
    pred = pd.read_csv(csv_path)

    # Step 3: Load model
    print("===== Checking Cellpose Model =====")
    try:
        model = models.Cellpose(device=device, model_type=args.model_type)
        print(f"[OK] Model {args.model_type} is ready")
    except Exception as e:
        print(f"[Error] Model loading failed: {e}")
        print("[Tip] Please check internet connection, Cellpose needs to download pretrained models")
        return
        
    # Step 4: Process each image
    for i, row in pred.iterrows():
        start_img = time.time()
        filename = row["Filenames"]
        name_stem = '.'.join(filename.split('.')[:-1])
        tif_path = os.path.join(IMAGE_DIR, f"{name_stem}.tif")

        print(f"\n===== Processing image: {filename} =====")

        if not os.path.exists(tif_path):
            print(f"Warning: Image file not found: {tif_path}, skipping")
            continue

        # --- Substep 4.1: Image loading and preprocessing ---
        try:
            enhanced_target, visualization_gray, H, W = load_and_preprocess_image(tif_path, args)
        except Exception as e:
            print(f"Error: Image preprocessing failed: {e}, skipping")
            continue

        # --- Substep 4.2: Full-image segmentation ---
        try:
            full_cell_mask = run_cellpose_on_full_image(model, enhanced_target, args)
        except Exception as e:
            print(f"Error: Full-image Cellpose prediction failed: {e}, skipping current image")
            continue

        # --- Substep 4.3: Load ROIs ---
        roi_masks = load_all_rois(region_ids, name_stem, ROI_DATA_DIR, H, W)
        if not roi_masks:
            print(f"Warning: No valid ROIs found, skipping image")
            continue

        # --- Substep 4.4: Statistical analysis ---
        result_updates, merged_mask = analyze_and_count_cells(full_cell_mask, roi_masks, H, W)
        
        # Assemble result dictionary
        result_dict = {
            "图像名称": filename,
            "图像高度(H)": H,
            "图像宽度(W)": W,
            "使用模型": args.model_type,
            "分析耗时(s)": round(time.time() - start_img, 2)
        }
        result_dict.update(result_updates)

        # --- Substep 4.5: Save visualizations ---
        try:
            save_visualizations(
                visualization_gray, full_cell_mask, roi_masks, merged_mask,
                result_dict, name_stem, IMAGE_DIR, args
            )
        except Exception as e:
            print(f"Warning: Visualization saving failed: {e}")

        all_results_list.append(result_dict)
        print(f"Image {filename} processed successfully")

    # --- Step 5: Save final results ---
    if all_results_list:
        save_final_results(all_results_list, IMAGE_DIR)
    else:
        print("\nWarning: No valid results to save")

if __name__ == "__main__":
    main()