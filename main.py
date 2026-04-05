import time
import os
import sys
import subprocess
import re
import argparse
from utils.config import load_config
from utils.region_selector import select_region
from modules.run_deepslice import run_deepslice
from modules.run_atlas import run_atlas
from modules.run_cellpose import run_cellpose

# 设置环境变量，强制子进程也用UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "zh_CN.UTF-8"
os.environ["LC_ALL"] = "zh_CN.UTF-8"

ENV_PACKAGES = {
    "deepslice": [
        "deepslice",
        "requests"
    ],
    "allensdk": [
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scipy==1.10.1",
        "h5py",
        "tables",
        "allensdk",
        "matplotlib"
    ],
    "cellpose": [
        "cellpose==3.0",
        "scikit-image",
        "numpy",
        "opencv-python-headless",
        "torch",
        "pandas",
        "matplotlib"
    ]
}

def run_conda_env_cmd(env_name, cmd, capture_output=True):
    """在指定conda环境中执行命令"""
    full_cmd = f"conda run -n {env_name} {cmd}"
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=capture_output, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        print(f"[错误] 执行命令 {full_cmd} 失败: {e}")
        return None

def parse_package_spec(spec_str):
    """解析包规范，返回 (包名, 版本规范)
    支持格式：numpy<2.0 / cellpose==3.0 / h5py
    """
    pattern = re.compile(r'^([a-zA-Z0-9_-]+)([<>=!~]+.*)$')
    match = pattern.match(spec_str.strip())
    if match:
        pkg_name = match.group(1).strip()
        version_spec = match.group(2).strip()
        from packaging.specifiers import SpecifierSet
        return pkg_name, SpecifierSet(version_spec)
    else:
        return spec_str.strip(), None

def check_package_version(env_name, package_spec):
    """检查指定conda环境中的包是否符合版本规范"""
    pkg_name, version_spec = parse_package_spec(package_spec)
    
    # 获取包的版本信息
    pip_show_output = run_conda_env_cmd(env_name, f"pip show {pkg_name}")
    if not pip_show_output:
        print(f"[缺失] 环境 {env_name} 中未找到包 {pkg_name}")
        return False
    
    # 解析版本号
    version_line = [line for line in pip_show_output.split('\n') if line.startswith('Version:')]
    if not version_line:
        print(f"[错误] 无法获取 {env_name} 中 {pkg_name} 的版本信息")
        return False
    pkg_version = version_line[0].split(':')[1].strip()
    
    # 检查版本是否符合要求
    if version_spec:
        from packaging.version import Version
        if Version(pkg_version) not in version_spec:
            print(f"[版本不匹配] 环境 {env_name} 中 {pkg_name} 版本为 {pkg_version}，不符合要求 {version_spec}")
            return False
        else:
            print(f"[OK] 环境 {env_name} 中 {pkg_name} 版本 {pkg_version} 符合要求 {version_spec}")
            return True
    else:
        print(f"[OK] 环境 {env_name} 中 {pkg_name} 已安装 (版本: {pkg_version})")
        return True

def install_package_to_conda_env(env_name, package_spec, pip_mirror="https://pypi.tuna.tsinghua.edu.cn/simple"):
    """安装/更新指定包到conda环境"""
    print(f"[安装] 正在为环境 {env_name} 安装/更新 {package_spec}...")
    install_cmd = f"pip install --upgrade {package_spec} -i {pip_mirror}"
    full_cmd = f"conda run -n {env_name} {install_cmd}"
    try:
        subprocess.run(
            full_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True
        )
        print(f"[OK] 环境 {env_name} 中 {package_spec} 安装/更新完成")
    except subprocess.CalledProcessError as e:
        print(f"[错误] 环境 {env_name} 中安装 {package_spec} 失败: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region-ids', type=str)
    parser.add_argument('--use-multithreading', action='store_true', 
                        help='Enable multithreading for faster processing (default: False)')
    parser.add_argument('--auto-diameter', default='False',choices=['True','False'],
                        help='Let Cellpose automatically estimate cell diameter')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Computation device (cpu/cuda/mps, default: cpu)')
    parser.add_argument('--save-masks', action='store_true', default=True,
                        help='Save predicted segmentation masks for debugging (default: True)')
    parser.add_argument('--enhance-contrast', action='store_true', default=True,
                        help='Enable CLAHE contrast enhancement for dim fluorescence images (default: True)')
    parser.add_argument('--model-type', type=str, default='cyto3', choices=['cyto3', 'cyto2', 'nuclei'],
                        help='Cellpose model type (nuclei for nuclear staining, cyto3/cyto2 for whole cell, default: cyto3)')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0,
                        help='Cell probability threshold, lower = more sensitive (-2 to -4 for weak signals, default: 0.0)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                        help='Flow threshold, lower = easier to split dense cells (0.3-0.5 recommended, default: 0.4)')
    parser.add_argument('--max-tile-edge', type=int, default=8000,
                        help='Maximum tile edge length for image splitting (default: 8000)')
    parser.add_argument('--clahe-clip', type=float, default=1.0,
                        help='CLAHE contrast limit, lower = softer (reduce if overexposed, default: 1.0)')
    parser.add_argument('--clahe-grid', type=int, default=16,
                        help='CLAHE grid size, larger = more global enhancement (default: 16)')
    parser.add_argument('--trim-percent', type=float, default=1.0, 
                        help='Percentile to trim from both ends for outlier suppression (default: 1.0)')
    return parser.parse_args()

def check_env():
    # 先检查packaging库（版本解析依赖）
    try:
        import packaging.version
        import packaging.specifiers
    except ImportError:
        print("[安装] 缺少版本解析依赖库packaging，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
            check=True, stdout=sys.stdout, stderr=sys.stderr, text=True
        )
        import packaging.version
        import packaging.specifiers

    import setup_env
    print("\n===== 开始检查运行环境 =====")

    # 1. 获取所有已存在的conda环境
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, check=True
        )
        env_list = result.stdout.lower()
    except Exception as e:
        print(f"[错误] 无法获取conda环境列表: {e}")
        raise

    # 2. 检查每个环境及其依赖包
    for env_name, packages in ENV_PACKAGES.items():
        print(f"\n----- 检查环境: {env_name} -----")
        if env_name not in env_list:
            print(f"[缺失] {env_name} 环境不存在，开始自动安装...")
            if env_name == "deepslice":
                setup_env.setup_deepslice()
            elif env_name == "allensdk":
                setup_env.setup_allensdk()
            elif env_name == "cellpose":
                setup_env.setup_cellpose()
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True, text=True, check=True
            )
            env_list = result.stdout.lower()
            if env_name not in env_list:
                raise RuntimeError(f"[错误] {env_name} 环境安装失败")
        else:
            print(f"[OK] {env_name} 环境已存在")

        missing_or_invalid = []
        for pkg_spec in packages:
            if not check_package_version(env_name, pkg_spec):
                missing_or_invalid.append(pkg_spec)
        
        # 4. 自动修复缺失/版本不匹配的包
        if missing_or_invalid:
            print(f"\n[更新] 环境 {env_name} 中有 {len(missing_or_invalid)} 个包需要修复: {missing_or_invalid}")
            for pkg_spec in missing_or_invalid:
                install_package_to_conda_env(env_name, pkg_spec)
        
        print(f"[完成] 环境 {env_name} 检查/更新完成")

    # 5. macOS系统依赖检查（HDF5）
    if sys.platform == 'darwin':
        setup_env.auto_install_brew_and_hdf5()

    print("\n===== 环境检查完成 =====")

def main():
    check_env()  # 启用环境检查
    cfg = load_config()
    args = parse_args()

    config = {
        "image_dir": cfg["image_dir"],
        "region_ids": list(map(str, args.region_ids)), 
        "cellpose": {
            "model_type": args.model_type,
            "device": args.device,
            "auto_diameter": args.auto_diameter == "True",
            "cellprob_threshold": args.cellprob_threshold,
            "flow_threshold": args.flow_threshold
        }
}

    run_pipeline(config)

def run_pipeline(config, log=print):
    IMAGE_DIR = config["image_dir"]
    region_ids = config['region_ids']

    start = time.time()

    log("STEP1 DeepSlice")
    run_deepslice(IMAGE_DIR)
    log(f"STEP2 Atlas ROI: {region_ids}")
    run_atlas(region_ids, IMAGE_DIR)
    log(f"STEP3 Cellpose: {region_ids}")
    run_cellpose(IMAGE_DIR, region_ids, config["cellpose"])
    
    log(f"Pipeline结束, 用时: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()