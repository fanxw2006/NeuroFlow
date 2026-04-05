import subprocess

def run_cellpose(image_dir, region_ids, params, merge_roi):
    cmd = [
        "conda", "run", "-n", "cellpose",
        "python", "cellpose_test.py",
        "--image-dir", image_dir,
        "--model-type", params["model_type"],
        "--device", params["device"],
        "--cellprob-threshold", str(params["cellprob_threshold"]),
        "--flow-threshold", str(params["flow_threshold"]),
        "--auto-diameter", str(params["auto_diameter"]),
        "--region-ids"
    ] + region_ids
    subprocess.run(cmd)