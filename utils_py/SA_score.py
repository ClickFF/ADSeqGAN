from __future__ import absolute_import, division, print_function
from builtins import range
# import organ
import os
import numpy as np
import pandas as pd
import csv
import time
import pickle
import gzip
import math
import random
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from copy import deepcopy
from math import exp, log
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

def readSAModel(filename='SA_score.pkl.gz'):
    print("mol_metrics: reading SA model ...")
    start = time.time()
    if filename == 'SA_score.pkl.gz':
        filename = os.path.join(os.path.dirname(__file__), filename)
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    end = time.time()
    print("loaded in {}".format(end - start))
    return SA_model


SA_model = readSAModel()

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

#===== Synthetics Accesability score ===


def SA_score(smile):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                        Chem.SANITIZE_SETAROMATICITY|
                        Chem.SANITIZE_SETCONJUGATION|
                        Chem.SANITIZE_SETHYBRIDIZATION|
                        Chem.SANITIZE_SYMMRINGS,
                    catchErrors=True)
        except:
            return 0
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    return val

if __name__ == "__main__":
    import os
    # 创建一个列表存储所有批次的统计信息
    all_stats = []
    
    # 基础路径
    # base_path = 'C:\\Users\\23163\\Desktop\\NA_GAN\\RESULTS\\ACseqGAN\\NAPro\\final\\CRC\\epoch_data'
    base_path = '/ihome/jwang/hat170/ACORGAN/epoch_data'
    save_path = '/ihome/jwang/hat170/ACORGAN/results'
    # if not os.path.exists(base_path):
    #     print(f"错误：基础路径不存在: {base_path}")
    #     exit(1)
    
    # 循环处理
    processed_files = 0
    for j in range(0, 2): 
        for i in range(0, 50):
            # 读取对应批次的文件
            input_file = os.path.join(base_path, f'NAPro_718_{i}_{j}.csv')
            # input_file = f'/home/hat170/Research/GAN_similarity/RESULTS/stdinrollout/None_Syn_{i}.smi'
            # input_file = os.path.join(base_path, f'Pro.csv')

            
            # 检查文件是否存在
            if not os.path.exists(input_file):
                print(f"跳过：文件不存在: {input_file}")
                continue
                
            print(f"\n正在处理文件: {input_file}")
            try:
                # 读取CSV文件
                # df = pd.read_csv(input_file, header=None, names=['smiles'], sep='\s+')
                df = pd.read_csv(input_file)
                processed_files += 1
                
                # 检查是否包含smiles列
                # if 'smiles' not in df.columns:
                #     print(f"错误：文件 {input_file} 中没有找到'smiles'列")
                #     continue
                
                # 确保文件不为空
                if df.empty:
                    print(f"警告: 文件 {input_file} 为空")
                    continue
                
                print(f"发现 {len(df)} 个分子")
                
                # 计算每个分子的SA值
                sa_values = []
                invalid_smiles = 0
                invalid_mols = 0
                sanitize_errors = 0
                
                for smiles in df['smiles']:
                    if not isinstance(smiles, str):
                        invalid_smiles += 1
                        continue
                    
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
                                sa_val = SA_score(mol)
                                sa_values.append(sa_val)
                            except:
                                sanitize_errors += 1
                                continue
                        else:
                            invalid_mols += 1
                    except:
                        invalid_smiles += 1
                        continue
                
                # 打印处理统计
                print(f"处理统计:")
                print(f"- 无效SMILES字符串: {invalid_smiles}")
                print(f"- 无效分子: {invalid_mols}")
                print(f"- Sanitize错误: {sanitize_errors}")
                print(f"- 成功计算SA: {len(sa_values)}")
                
                # 计算统计信息
                if sa_values:
                    avg_sa = np.mean(sa_values)
                    std_sa = np.std(sa_values)
                    total_molecules = len(df)
                    valid_sa_count = len(sa_values)
                    
                    # 收集统计信息
                    stats = {
                        'batch_id': i,
                        'classifier_id': j,
                        'total_molecules': total_molecules,
                        'valid_sa_count': valid_sa_count,
                        'invalid_smiles': invalid_smiles,
                        'invalid_mols': invalid_mols,
                        'sanitize_errors': sanitize_errors,
                        'average_sa': round(avg_sa, 4),
                        'std_sa': round(std_sa, 4)
                    }
                    all_stats.append(stats)
                    
                    # 打印当前批次统计信息
                    print(f"\n批次 {i}_{j} 统计信息:")
                    print(f"总分子数: {total_molecules}")
                    print(f"有效SA计算数: {valid_sa_count}")
                    print(f"平均SA值: {avg_sa:.4f}")
                    print(f"SA标准差: {std_sa:.4f}")
                    
                    # 将SA值添加到数据框并保存
                    # df['sa'] = pd.Series(sa_values + [None] * (len(df) - len(sa_values)))
                    # output_file = os.path.join(base_path, f'stdinrollout_{i}_{j}_sa.csv')
                    # df.to_csv(output_file, index=False)
                    # print(f"结果已保存至: {output_file}")
                
            except Exception as e:
                print(f"处理批次 {i}_{j} 时出错: {str(e)}")
                continue
    
    print(f"\n总共处理了 {processed_files} 个文件")
    
    # 将所有统计信息保存到汇总文件
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_output_file = os.path.join(save_path, 'NAPro_718_sa_statistics_summary.csv')
        stats_df.to_csv(stats_output_file, index=False)
        print(f"\nSA统计汇总已保存至: {stats_output_file}")
    else:
        print("\n警告：没有成功处理任何批次的数据")
