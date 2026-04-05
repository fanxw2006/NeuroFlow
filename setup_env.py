import subprocess
import sys
import os
import platform

CONDA_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
CONDA_FORGE_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/"
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

def run_cmd(cmd, shell=True, check=True, capture_output=False, live_output=True):
    print(f"[执行] {cmd}")
    if live_output and not capture_output:
        return subprocess.run(
            cmd, shell=shell, check=check,
            stdout=sys.stdout, stderr=sys.stderr, text=True
        )
    else:
        return subprocess.run(
            cmd, shell=shell, check=check,
            capture_output=capture_output, text=True
        )

def check_brew_exists():
    """检查 Homebrew 是否真的安装了"""
    try:
        result = run_cmd("which brew", check=True, capture_output=True, live_output=False)
        return result.stdout.strip() != ""
    except:
        return False

def auto_install_brew_and_hdf5():
    """macOS专属：最稳健版本，不瞎折腾"""
    if sys.platform != 'darwin':
        return
    
    print("\n===== [macOS] 检查系统依赖 =====")
    
    # 1. 先检查 Homebrew 是否存在
    if not check_brew_exists():
        print("❌ [错误] 未检测到 Homebrew")
        print("   请先在终端运行以下命令安装 Homebrew，然后再运行此程序：")
        print('   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        print("   安装完成后，请重启终端。")
        # 不报错退出，只是提示，让程序继续尝试
        return
    
    print("[OK] Homebrew 已检测到")
    
    # 2. 检查 HDF5 是否安装，没装就装
    try:
        run_cmd("brew list hdf5", check=True, capture_output=True, live_output=False)
        print("[OK] HDF5 已安装")
    except subprocess.CalledProcessError:
        print("[安装] 正在安装 HDF5 (这可能需要几分钟)...")
        try:
            run_cmd("brew install hdf5", check=True)
        except Exception as e:
            print(f"[警告] HDF5 安装可能遇到问题: {e}")
            print("   程序将继续尝试运行...")
    
    # 3. 设置环境变量
    if platform.machine() == 'arm64':
        hdf5_path = "/opt/homebrew/opt/hdf5"
    else:
        hdf5_path = "/usr/local/opt/hdf5"
    
    os.environ["HDF5_DIR"] = hdf5_path
    os.environ["CPPFLAGS"] = f"-I{hdf5_path}/include"
    os.environ["LDFLAGS"] = f"-L{hdf5_path}/lib"
    
    print(f"[OK] HDF5 环境变量已设置")
    print("===== 系统依赖检查完成 =====")

def create_env(name, packages):
    print(f"\n===== 正在处理环境: {name} =====")
    
    try:
        run_cmd(
            f"conda create -n {name} python=3.9 -y -c {CONDA_MIRROR} -c {CONDA_FORGE_MIRROR} --override-channels",
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"[提示] 环境 {name} 已存在，跳过创建")
    
    run_cmd(
        f"conda run -n {name} pip install --upgrade pip setuptools wheel -i {PIP_MIRROR}",
        check=True
    )
    
    if packages:
        safe_packages = []
        for pkg in packages:
            if '<' in pkg or '>' in pkg or '=' in pkg:
                safe_packages.append(f'"{pkg}"')
            else:
                safe_packages.append(pkg)
        
        pkg_str = " ".join(safe_packages)
        run_cmd(
            f"conda run -n {name} pip install {pkg_str} -i {PIP_MIRROR}",
            check=True
        )
        print(f"[OK] 环境 {name} 依赖安装完成")

def setup_deepslice():
    create_env(
        name="deepslice",
        packages=["deepslice", "requests"]
    )

def setup_allensdk():
    auto_install_brew_and_hdf5()
    create_env(
        name="allensdk",
        packages=["numpy==1.23.5", "pandas==1.5.3", "scipy==1.10.1", "h5py", "tables", "allensdk", "matplotlib"]
    )

def setup_cellpose():
    create_env(
        name="cellpose",
        packages=["cellpose==3.0", "scikit-image", "numpy", "opencv-python-headless", "torch", "pandas", "matplotlib"]
    )

def setup_all():
    setup_deepslice()
    setup_allensdk()
    setup_cellpose()

if __name__ == "__main__":
    setup_all()