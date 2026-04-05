import os
import requests
import h5py
from DeepSlice import DSModel
import importlib.util
import argparse

WEIGHTS = [
    ("xception_weights_tf_dim_ordering_tf_kernels.h5", "https://data-proxy.ebrains.eu/api/v1/buckets/deepslice/weights/xception_weights_tf_dim_ordering_tf_kernels.h5"),
    ("Allen_Mixed_Best.h5", "https://data-proxy.ebrains.eu/api/v1/buckets/deepslice/weights/Allen_Mixed_Best.h5")
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    return parser.parse_args()

def get_deepslice_weights_dir():
    spec = importlib.util.find_spec("DeepSlice")
    if spec is None:
        raise ImportError("DeepSlice package not found")
    package_root = os.path.dirname(spec.origin)
    weights_dir = os.path.join(package_root, "metadata", "weights")
    return weights_dir

def is_hdf5_valid(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, 'r') as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False

def check_all_weights(WEIGHTS_DIR):
    for filename, _ in WEIGHTS:
        fpath = os.path.join(WEIGHTS_DIR, filename)
        if not is_hdf5_valid(fpath):
            return False
    return True

def download_weights_manually(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for filename, url in WEIGHTS:
        fpath = os.path.join(WEIGHTS_DIR, filename)
        
        if os.path.exists(fpath):
            os.remove(fpath)
            
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(fpath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Manual download failed for {filename}")

def main():
    args = parse_args()

    WEIGHTS_DIR = get_deepslice_weights_dir()
    print(f"Auto-detected weights dir: {WEIGHTS_DIR}")
    
    if check_all_weights(WEIGHTS_DIR):
        print("Weight downloaded successful.")
    else:
        print("Attempting automatic download...")
        try:
            model = DSModel('mouse')
            if check_all_weights(WEIGHTS_DIR):
                print("Automatic download successful.")
            else:
                print("Automatic download failed.")
        except Exception:
            print("Automatic download failed.")

        if not check_all_weights(WEIGHTS_DIR):
            print("Attempting manual download...")
            try:
                download_weights_manually(WEIGHTS_DIR)
                if check_all_weights(WEIGHTS_DIR):
                    print("Manual download successful.")
                else:
                    print("Manual download failed.")
                    raise RuntimeError("Manual download produced invalid files.")
            except Exception as e:
                print(f"Manual download failed: {e}")
                return
    
    model = DSModel('mouse')
    
    folderpath = str(args.image_dir)
    model.predict(folderpath, ensemble=False, section_numbers=False)
    save_path = os.path.join(folderpath, 'MyResults')
    model.save_predictions(save_path)
    print("DeepSlice 运行完成！")

if __name__ == "__main__":
    main()