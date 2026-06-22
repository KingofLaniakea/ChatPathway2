import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

def download_qwen3_base():
    REPO_ID = "Qwen/Qwen3-8B"
    TARGET_DIR = "/root/autodl-tmp/qwen3_8b_base"
    # ==============================================================

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📂 已成功创建数据盘目标文件夹: {TARGET_DIR}")

    try:
        print(" 正在开启 16 线程国内镜像专属高速下载通道（支持断点续传）...")
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,  # 🌟 设为 False，直接下载真实文件，防止软链接在移动时失效
            repo_type="model",
            max_workers=16                 # 16线程并发压榨网络带宽
        )
        print("\n================================================================")
        print("[Qwen3-8B 官方底座下载成功!]")
        print(f" 落地路径: '{TARGET_DIR}'")
        print("================================================================")
    except Exception as e:
        print("\n [下载发生意外中断]:")
        print(str(e))
        print("\n 提示: AutoDL 网络偶发抖动是正常现象。本脚本支持【断点续传】，直接再次运行即可继续增量下载！")

if __name__ == "__main__":
    download_qwen3_base()