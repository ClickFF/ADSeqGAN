from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib
from multiprocessing import Pool, cpu_count
from itertools import product
matplotlib.use('Agg')  # 使用非交互式后端

def calculate_fingerprint(mol, fp_type='ECFP4'):
    """
    计算分子指纹
    fp_type: 'ECFP4' 或 'MACCS'
    """
    try:
        if mol is None:
            return None
        if fp_type == 'ECFP4':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)  # ECFP4
        elif fp_type == 'MACCS':
            return AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"不支持的指纹类型: {fp_type}")
    except Exception as e:
        print(f"计算分子指纹时出错: {str(e)}")
        return None

def calculate_similarity_chunk(args):
    """
    计算相似性矩阵的一个块
    """
    start_idx, chunk_size, fps1, fps2 = args
    chunk_matrix = np.zeros((chunk_size, len(fps2)))
    for i in range(chunk_size):
        idx = start_idx + i
        if idx >= len(fps1):
            break
        for j in range(len(fps2)):
            chunk_matrix[i, j] = TanimotoSimilarity(fps1[idx], fps2[j])
    return chunk_matrix

def calculate_and_save_similarity_matrix(fps, chunk_size=1000, n_jobs=None):
    """
    分块计算相似性矩阵并直接保存到文件，降低内存使用
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    n_molecules = len(fps)
    n_chunks = (n_molecules + chunk_size - 1) // chunk_size

    # 创建临时文件来存储结果
    temp_file = 'temp_similarity_matrix.npy'
    
    # 分块计算并保存
    print(f"Starting parallel computation using {n_jobs} processes...")
    print(f"Total chunks: {n_chunks}")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, n_molecules - start_idx)
        
        # 准备当前块的任务
        sub_chunk_size = max(1, current_chunk_size // n_jobs)
        tasks = []
        for sub_start_idx in range(start_idx, start_idx + current_chunk_size, sub_chunk_size):
            remaining = min(sub_chunk_size, start_idx + current_chunk_size - sub_start_idx)
            tasks.append((sub_start_idx, remaining, fps, fps))
        
        # 并行计算当前块
        with Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap(calculate_similarity_chunk, tasks),
                              total=len(tasks),
                              desc=f"Computing chunk {chunk_idx + 1}/{n_chunks}"))
        
        # 组装当前块的结果
        chunk_matrix = np.vstack(results)
        
        # 保存或追加到文件
        if chunk_idx == 0:
            np.save(temp_file, chunk_matrix)
        else:
            # 加载现有数据并追加新的块
            existing_matrix = np.load(temp_file)
            combined_matrix = np.vstack([existing_matrix, chunk_matrix])
            np.save(temp_file, combined_matrix)
            del existing_matrix  # 释放内存
        
        del chunk_matrix  # 释放内存
    
    return temp_file

def plot_similarity_heatmap(matrix_file, output_file, max_display_size=2000):
    """
    绘制相似性热力图，支持大矩阵的降采样
    """
    # 加载数据
    similarity_matrix = np.load(matrix_file)
    
    # 如果矩阵太大，进行降采样
    if similarity_matrix.shape[0] > max_display_size:
        step = similarity_matrix.shape[0] // max_display_size
        similarity_matrix = similarity_matrix[::step, ::step]
        print(f"Matrix downsampled to {similarity_matrix.shape[0]}x{similarity_matrix.shape[0]} for visualization")
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, cmap='YlOrRd', square=True)
    plt.title('Tanimoto Similarity Matrix')
    plt.xlabel('Generated Molecules')
    plt.ylabel('Generated Molecules')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    # 基础路径
    data_path = os.getcwd()  # 使用当前工作目录
    
    # 收集所有生成的分子
    all_gen_fps = []
    total_invalid = 0
    
    print("Processing all NAPro files...")
    # 处理所有匹配的文件
    for filename in os.listdir(data_path):
        if filename.startswith('NAPro_') and filename.endswith('_40_0.csv'):
            gen_file = os.path.join(data_path, filename)
            print(f"\nProcessing file: {gen_file}")
            
            try:
                gen_df = pd.read_csv(gen_file)
                
                # 处理生成的分子
                print(f"Processing molecules from {filename}...")
                for smiles in tqdm(gen_df['smiles']):
                    try:
                        mol = Chem.MolFromSmiles(smiles, sanitize=False)
                        if mol is not None:
                            try:
                                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                                            Chem.SANITIZE_SETAROMATICITY|
                                            Chem.SANITIZE_SETCONJUGATION|
                                            Chem.SANITIZE_SETHYBRIDIZATION|
                                            Chem.SANITIZE_SYMMRINGS,
                                        catchErrors=True)
                                fp = calculate_fingerprint(mol, 'ECFP4')
                                if fp is not None:
                                    all_gen_fps.append(fp)
                                else:
                                    total_invalid += 1
                            except:
                                total_invalid += 1
                                continue
                        else:
                            total_invalid += 1
                    except:
                        total_invalid += 1
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
    
    print(f"\nTotal valid molecules collected: {len(all_gen_fps)}")
    print(f"Total invalid molecules: {total_invalid}")
    
    if len(all_gen_fps) > 0:
        # 使用分块计算相似性矩阵
        print("Calculating similarity matrix using chunked parallel processing...")
        matrix_file = calculate_and_save_similarity_matrix(all_gen_fps, chunk_size=1000, n_jobs=48)
        
        # 绘制热力图
        plot_similarity_heatmap(matrix_file, 'similarity_heatmap_all.png')
        print("\nHeatmap saved as 'similarity_heatmap_all.png'")
        
        # 清理临时文件
        os.remove(matrix_file)

