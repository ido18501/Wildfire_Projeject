#!/usr/bin/env python3
"""
Download script for Sim2Real-Fire dataset from Google Drive.

Run locally:
    pip install gdown
    python download_sim2real.py --output_dir ~/WFP/sim2real_fire_data/simulation_data

This will download all 20 simulation packages + real-world data.
For a quick test, use: python download_sim2real.py --packages 1 --output_dir ~/WFP/sim2real_fire_data/simulation_data
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Google Drive folder IDs for each simulation package
SIMULATION_PACKAGES = {
    1: ("1uSzsHcjtZeZvW76SSatxR5r2ps5KizEJ", "0001_02614 to 0019_01316"),
    2: ("18QY858nhxsnsa1ZytIaVkuhQZgIN7jGj", "0020_00970 to 0046_02758"),
    3: ("1OdNSsGkB4ZpPEDSbSi6fQbQ0IbcpVTiy", "0047_05424 to 0068_03539"),
    4: ("1hR8mvVKCpswmYtsH7BR8PiZHaICJ8Apb", "0069_03539 to 0077_11679"),
    5: ("1enqv5xdJ_fsBpt5e7wCt9R-8JOBPhFaP", "0078_14685 to 0084_02609"),
    6: ("1CauT4UIMy0Ckc4uTZFyQ2v9XMtvewKcD", "0085_02466 to 0114_02292"),
    7: ("1nMJ7VS72EbANUVRzpFxF4q4XgDBTvt_6", "0115_02715 to 0130_01761"),
    8: ("1G8zZAuebZIlo_jB_3I5RZGIDvXK50k6P", "0131_02592 to 0152_02969"),
    9: ("1kQlamHitO_B-hK3KGHjLIfFO_kiu1iWS", "0153_00235 to 0168_03137"),
    10: ("1j7Xi71W7HL1FrredUc-g6CMVTTb3Us2o", "0169_23343 to 0177_01629"),
    11: ("137Ix90KidXb4M_Sh73te9ExAHOhMGWyd", "0178_08498 to 0181_07922"),
    12: ("1k2InB5J_d7iNA-jelfgDi88y3dgeqZnJ", "0182_25232 to 0198_00609"),
    13: ("1n9kwjp7Ve-4-IXE8tYluRe-k8Uh1ZodZ", "0199_13351 to 0213_01983"),
    14: ("1sJLoQfHnCACbgnGiYn7Uj9MN6FJuTah6", "0214_02091 to 0242_02940"),
    15: ("1s-YQ6mHinZ6wmD_BBCfL1xhJs3gUdzti", "0243_02722 to 0260_03391"),
    16: ("1JLB2QXaxUmq-OEm01v5omHPtzt5fw0Yw", "0261_02900 to 0282_03685"),
    17: ("1vJY3gkwuO7J_AyMxDTI4OELnMrYpXEoq", "0283_00606 to 0310_02903"),
    18: ("1yYPsRzQy9OWBSZWJVmVldjp__FjUe1Tl", "0311_01395 to 0328_02411"),
    19: ("1W2vtrZmfu_L9tvB4ro4OL1Fjka18fzOS", "0329_02902 to 0349_03066"),
    20: ("1AZHn5a8GraWjGlbGDX3fRwibqJ0vOaiO", "0350_03311 to 0351_00343"),
}

REALWORLD_DATA = ("1QJbNu35VzCCDOhkc3bn5guIoLEyA0cRF", "real-world wildfire data")


def check_gdown():
    """Check if gdown is installed."""
    try:
        import gdown
        return True
    except ImportError:
        print("gdown not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True


def download_folder(folder_id: str, output_dir: Path, description: str):
    """Download a Google Drive folder using gdown."""
    import gdown
    
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"\n{'='*70}")
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    try:
        gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
        return True
    except Exception as e:
        print(f"Error downloading {description}: {e}")
        print("Try downloading manually from the URL above.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Sim2Real-Fire dataset")
    parser.add_argument("--output_dir", type=str, default="./sim2real_fire_data/simulation_data",
                       help="Output directory for simulation data")
    parser.add_argument("--packages", type=str, default="all",
                       help="Which packages to download: 'all', or comma-separated numbers like '1,2,3'")
    parser.add_argument("--include_realworld", action="store_true",
                       help="Also download real-world data")
    parser.add_argument("--realworld_dir", type=str, default="./sim2real_fire_data/realworld_data",
                       help="Output directory for real-world data")
    args = parser.parse_args()
    
    # Check gdown
    check_gdown()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse packages to download
    if args.packages == "all":
        packages_to_download = list(SIMULATION_PACKAGES.keys())
    else:
        packages_to_download = [int(p.strip()) for p in args.packages.split(",")]
    
    print(f"\n{'#'*70}")
    print(f"# Sim2Real-Fire Dataset Downloader")
    print(f"# Packages: {packages_to_download}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*70}")
    
    # Download simulation packages
    success_count = 0
    for pkg_num in packages_to_download:
        if pkg_num not in SIMULATION_PACKAGES:
            print(f"Warning: Package {pkg_num} not found, skipping.")
            continue
        
        folder_id, description = SIMULATION_PACKAGES[pkg_num]
        if download_folder(folder_id, output_dir, f"simulation_data_{pkg_num:02d} ({description})"):
            success_count += 1
    
    # Download real-world data if requested
    if args.include_realworld:
        realworld_dir = Path(args.realworld_dir)
        realworld_dir.mkdir(parents=True, exist_ok=True)
        folder_id, description = REALWORLD_DATA
        download_folder(folder_id, realworld_dir, description)
    
    print(f"\n{'='*70}")
    print(f"Download complete: {success_count}/{len(packages_to_download)} packages")
    print(f"Data location: {output_dir}")
    print("="*70)
    
    # Print summary of what should be there
    print("\nExpected folder structure:")
    print(f"  {output_dir}/")
    print("    ├── 0001_02614/")
    print("    ├── 0002_xxxxx/")
    print("    ├── ...")
    print("    └── 0351_00343/")


if __name__ == "__main__":
    main()
