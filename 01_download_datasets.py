import os
import requests
from tqdm import tqdm

def download_geo_file(url, save_path):
    """带进度条的远程流式下载函数"""
    if os.path.exists(save_path):
        print(f"[跳过] 文件已存在: {save_path}")
        return
    
    print(f"[下载中] 正在拉取: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status() # 如果网络出错直接抛出异常
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(save_path)
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    print(f"[成功] 已保存至: {save_path}\n")

if __name__ == "__main__":

    data_dir = "./L1000_GSE92743_Data"
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92743/suppl"
    
    files_to_download = {
        # "matrix": f"{base_url}/GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_n3176x12320.gctx.gz",
        "gene_info": f"{base_url}/GSE92743_Broad_GTEx_gene_info.txt.gz",
        # "inst_info": f"{base_url}/GSE92743_Broad_GTEx_inst_info.txt.gz"
    }

    for file_key, url in files_to_download.items():
        filename = url.split("/")[-1]
        target_path = os.path.join(data_dir, filename)
        try:
            download_geo_file(url, target_path)
        except Exception as e:
            print(f"[错误] 下载 {filename} 失败，原因: {e}")