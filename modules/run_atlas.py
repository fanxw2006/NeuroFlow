import subprocess

def run_atlas(region_ids, image_dir):
    cmd = [
        "conda", "run", "-n", "allensdk",
        "python", "allensdk_ROI.py",
        "--image-dir", image_dir,
        "--region-ids"
    ] + region_ids
    subprocess.run(cmd)