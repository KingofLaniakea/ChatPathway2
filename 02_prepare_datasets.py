# import os
# import gzip
# import shutil

# # 定义你的数据存放目录
# data_dir = "./L1000_GSE92743_Data"

# files_to_unpack = {
#     "gene_info": "GSE92743_Broad_GTEx_gene_info.txt.gz",
#     "inst_info": "GSE92743_Broad_GTEx_inst_info.txt.gz",
#     "matrix": "GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_n3176x12320.gctx.gz"
# }

# print("="*50)
# print("开始对 L1000 GSE92743 数据集进行完整解压...")
# print("="*50)

# for key, filename in files_to_unpack.items():
#     src_path = os.path.join(data_dir, filename)
    
#     # 如果用户已经提前手动解压了，或者是换了名字，做个检查
#     if not os.path.exists(src_path):
#         print(f"[警告] 找不到压缩包: {src_path}，跳过此文件。")
#         continue
    
#     # 移除最后的 .gz 作为解压后的新文件名
#     dest_path = src_path.replace(".gz", "")
    
#     print(f"正在解压: {filename} \n -> 至: {os.path.basename(dest_path)} ...")
    
#     try:
#         with gzip.open(src_path, 'rb') as f_in:
#             with open(dest_path, 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)
#         print(f"[成功] 解压完成！\n")
#     except Exception as e:
#         print(f"[失败] 解压 {filename} 出错: {e}\n")

# print("="*50)
# print("所有文件解压流程结束！")
# print("="*50)

# import pandas as pd
# import gzip

# file_path = "./L1000_GSE92743_Data/GSE92743_Broad_GTEx_gene_info.txt"

# print("="*60)
# print("正在尝试强行读取文件内容...")
# print("="*60)

# # 机制一：如果它只是个改了名字的 .gz 压缩包，用 gzip 模块直接读
# try:
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         df = pd.read_csv(f, sep="\t", nrows=5)
#     print("[成功通过机制一读取] 文件本质确实还是压缩包！")
#     print("\n表格列名 (Columns):", df.columns.tolist())
#     print("\n前 3 行数据快照:")
#     print(df.head(3))
# except Exception as e:
#     print(f"[机制一失败] 无法用 Gzip 引擎解析: {e}")
    
#     # 机制二：如果它是个坏了编码的普通文本，强行忽略损坏的二进制字节读前几行
#     try:
#         df = pd.read_csv(file_path, sep="\t", nrows=5, encoding='utf-8', encoding_errors='ignore')
#         print("\n[成功通过机制二读取] 已忽略损坏的二进制字节！")
#         print("\n表格列名 (Columns):", df.columns.tolist())
#         print("\n前 3 行数据快照:")
#         print(df.head(3))
#     except Exception as e2:
#         print(f"[机制二失败] 依然无法用 pandas 读入: {e2}")
        
#         # 机制三：最底层的物理肉眼模式，直接当成纯二进制字节，打印前 500 个字符看看它到底是个啥
#         print("\n[启动底层二进制探查] 打印文件前 500 个原始字符：")
#         with open(file_path, 'rb') as f:
#             raw_bytes = f.read(500)
#             print(raw_bytes)

# import pandas as pd

# # 读取我们刚刚解压出来的 cleaned 文件
# cleaned_gene_path = "./L1000_GSE92743_Data/GSE92743_Broad_GTEx_gene_info.txt"

# df_gene = pd.read_csv(cleaned_gene_path, sep="\t")
# print("【验证成功】干净的基因表格列名:", df_gene.columns.tolist())
# print("\n前 3 行正常数据:")
# print(df_gene.head(3))


from cmapPy.pandasGEXpress.parse_gctx import parse

matrix_path = "./L1000_GSE92743_Data/GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_n3176x12320.gctx"

# 1. 拿到所有的行和列的唯一 ID 列表
# （我们需要用它的底层工具 read_rid 和 read_cid 快速摸一下索引）
import h5py
with h5py.File(matrix_path, 'r') as f:
    all_row_ids = [x.decode('utf-8') for x in f['0']['META']['ROW']['id'][:]]
    all_col_ids = [x.decode('utf-8') for x in f['0']['META']['COL']['id'][:]]

# 2. 精准指定：我只切片读取前 5 个样本和前 5 个基因
sample_rid = all_row_ids[:5]
sample_cid = all_col_ids[:5]

# 3. 使用 cmapPy 进行局部解析
gct_sliced = parse(matrix_path, rid=sample_rid, cid=sample_cid)
df_slice = gct_sliced.data_df

print("\n" + "="*60)
print("【左上角 5x5 表达量数值矩阵快照】")
print("="*60)
print(df_slice)