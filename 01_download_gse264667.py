import os
import requests
from tqdm import tqdm

def download_geo_file(url, save_path):
    if os.path.exists(save_path):
        print(f"[ SKIP ] File already exists: {save_path}")
        return
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status() 
    
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024 
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            if data:
                f.write(data)
                bar.update(len(data))
    print(f"[ SUCCESS ] Saved to: {save_path}\n")

if __name__ == "__main__":

    data_dir = "./CRISPR_GSE264667_Data"
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl"
    
    files_to_download = {
        "jurkat_cell": f"{base_url}/GSE264667_jurkat_raw_singlecell_01.h5ad"
    }

    for file_key, url in files_to_download.items():
        filename = url.split("/")[-1]
        target_path = os.path.join(data_dir, filename)
        try:
            download_geo_file(url, target_path)
        except Exception as e:
            print(f"[ ERROR ] Failed to download {filename}. Reason: {e}")