import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="遞迴掃描指定資料夾，並印出所有 .npy 檔案的 shape"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="要掃描的根目錄路徑 (例如: Best_Video_Features)",
    )
    args = parser.parse_args()

    input_dir = args.input

    if not os.path.isdir(input_dir):
        print(f"找不到目錄 '{input_dir}'")
        return

    print(f"開始掃描 '{input_dir}' 及其子資料夾底下的 .npy 檔案...\n")

    # 尋找所有 .npy 檔案 (包含所有子資料夾)
    search_pattern = os.path.join(input_dir, "**", "*.npy")
    npy_files = glob.glob(search_pattern, recursive=True)

    if not npy_files:
        print("沒有找到任何 .npy 檔案。")
        return

    # 將結果排序，讓輸出更容易閱讀
    npy_files.sort()

    for npy_file in npy_files:
        # 計算相對路徑以便顯示，這樣畫面不會被超長絕對路徑佔滿
        rel_path = os.path.relpath(npy_file, input_dir)
        try:
            loaded = np.load(npy_file)
            print(f"{rel_path} -> shape: {loaded.shape}")
        except Exception as e:
            print(f"無法讀取 {rel_path}: {e}")

    print(f"\n共找到 {len(npy_files)} 個 .npy 檔案。")


if __name__ == "__main__":
    main()
