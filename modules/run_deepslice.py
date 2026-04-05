import subprocess
import tempfile
import os
def run_deepslice(image_dir):
 with tempfile.TemporaryDirectory() as tmpdir:
    env = os.environ.copy()
    env["TMP"] = tmpdir
    env["TEMP"] = tmpdir
    subprocess.run([
        "conda","run","-n","deepslice",
        "python","deepslice.py",
        "--image-dir", image_dir
    ])