import os
import numpy as np
import cv2
import pandas as pd
from skimage.measure import label
from typing import Tuple, List, Dict, Any

def load_all_rois(
    region_ids: List[int], 
    name_stem: str, 
    ROI_DATA_DIR: str, 
    H: int, 
    W: int
) -> Dict[int, np.ndarray]:
    """Load all specified ROI masks.
    
    Args:
        region_ids: List of ROI IDs to load.
        name_stem: Image filename stem.
        ROI_DATA_DIR: Directory containing ROI data.
        H: Target image height.
        W: Target image width.
        
    Returns:
        dict: Mapping from ROI ID to binary mask.
    """
    roi_masks = {}
    for rid in region_ids:
        color_csv_path = os.path.join(ROI_DATA_DIR, f"{name_stem}_color_mapping.csv")
        roi_mask_path = os.path.join(ROI_DATA_DIR, f"{name_stem}_ROI_mask.png")

        if not os.path.exists(color_csv_path) or not os.path.exists(roi_mask_path):
            print(f"Warning: ROI files missing for ID {rid}, skipping")
            continue

        try:
            df = pd.read_csv(color_csv_path)
            color_map = {int(row.target_id): (int(row.R), int(row.G), int(row.B)) for _, row in df.iterrows()}
            if rid not in color_map and str(rid) not in color_map:
                print(f"Warning: ROI ID {rid} not in mapping table")
                continue
            
            target_rgb = color_map.get(rid, color_map.get(str(rid)))
            roi_color_mask = cv2.imread(roi_mask_path, cv2.IMREAD_COLOR)

            if roi_color_mask.shape[:2] != (H, W):
                roi_color_mask = cv2.resize(roi_color_mask, (W, H), interpolation=cv2.INTER_NEAREST)

            color = (target_rgb[2], target_rgb[1], target_rgb[0]) # b, g, r
            roi_masks[rid] = cv2.inRange(roi_color_mask, color, color)

        except Exception as e:
            print(f"Error: ROI {rid} extraction failed: {e}")
            continue
    return roi_masks

def analyze_and_count_cells(
    full_cell_mask: np.ndarray, 
    roi_masks: Dict[int, np.ndarray], 
    H: int, 
    W: int
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Count cells in full image, individual ROIs, and merged ROI.
    
    Args:
        full_cell_mask: Full segmentation mask.
        roi_masks: Dictionary of ROI masks.
        H: Image height.
        W: Image width.
        
    Returns:
        tuple: (result_updates dict, merged_mask array)
    """
    result_updates = {}
    result_updates["全图总细胞数"] = len(np.unique(full_cell_mask)) - 1

    for rid, roi_mask_binary in roi_masks.items():
        roi_area = np.sum(roi_mask_binary // 255)
        roi_cells_mask = full_cell_mask * (roi_mask_binary // 255)
        cell_count = len(np.unique(label(roi_cells_mask))) - 1
        result_updates[f"ROI_{rid}_面积"] = roi_area
        result_updates[f"ROI_{rid}_细胞数"] = cell_count

    merged_mask = np.zeros((H, W), dtype=np.uint8)
    for roi_mask_binary in roi_masks.values():
        merged_mask = cv2.bitwise_or(merged_mask, roi_mask_binary)
    
    merged_area = np.sum(merged_mask // 255)
    merged_cells_mask = full_cell_mask * (merged_mask // 255)
    merged_cell_count = len(np.unique(label(merged_cells_mask))) - 1
    result_updates["ROI_总面积"] = merged_area
    result_updates["ROI_总细胞数"] = merged_cell_count
    print(f"Analysis complete: Merged region cell count = {merged_cell_count}")

    return result_updates, merged_mask