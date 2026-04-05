import subprocess

def run_atlas(region_ids, image_dir, merge_roi):
    cmd = [
        "conda", "run", "-n", "allensdk",
        "python", "allensdk_ROI.py",
        "--image-dir", image_dir,
        "--region-ids"
    ] + region_ids
    subprocess.run(cmd)