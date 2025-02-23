from __future__ import absolute_import, division, print_function
from builtins import range
# import organ
import os
import numpy as np
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from copy import deepcopy
from math import exp, log
import pandas as pd
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

#====== qed variables

AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')

AcceptorSmarts = [
    '[oH0;X2]',
    '[OH1;X2;v2]',
    '[OH0;X2;v2]',
    '[OH0;X1;v2]',
    '[O-;X1]',
    '[SH0;X2;v2]',
    '[SH0;X1;v2]',
    '[S-;X1]',
    '[nH0;X2]',
    '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))

StructuralAlertSmarts = [
    '*1[O,S,N]*1',
    '[S,C](=[O,S])[F,Br,Cl,I]',
    '[CX4][Cl,Br,I]',
    '[C,c]S(=O)(=O)O[C,c]',
    '[$([CH]),$(CC)]#CC(=O)[C,c]',
    '[$([CH]),$(CC)]#CC(=O)O[C,c]',
    'n[OH]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
    'C=C(C=O)C=O',
    'n1c([F,Cl,Br,I])cccc1',
    '[CH1](=O)',
    '[O,o][O,o]',
    '[C;!R]=[N;!R]',
    '[N!R]=[N!R]',
    '[#6](=O)[#6](=O)',
    '[S,s][S,s]',
    '[N,n][NH2]',
    'C(=O)N[NH2]',
    '[C,c]=S',
    '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
    'C1(=[O,N])C=CC(=[O,N])C=C1',
    'C1(=[O,N])C(=[O,N])C=CC=C1',
    'a21aa3a(aa1aaaa2)aaaa3',
    'a31a(a2a(aa1)aaaa2)aaaa3',
    'a1aa2a3a(a1)A=AA=A3=AA=A2',
    'c1cc([NH2])ccc1',
    '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
    'I',
    'OS(=O)(=O)[O-]',
    '[N+](=O)[O-]',
    'C(=O)N[OH]',
    'C1NC(=O)NC(=O)1',
    '[SH]',
    '[S-]',
    'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
    'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
    '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
    '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
    '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
    '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    'C#C',
    '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
    '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
    '[C,c]=N[OH]',
    '[C,c]=NOC=O',
    '[C,c](=O)[CX4,CR0X3,O][C,c](=O)',
    'c1ccc2c(c1)ccc(=O)o2',
    '[O+,o+,S+,s+]',
    'N=C=O',
    '[NX3,NX4][F,Cl,Br,I]',
    'c1ccccc1OC(=O)[#6]',
    '[CR0]=[CR0][CR0]=[CR0]',
    '[C+,c+,C-,c-]',
    'N=[N+]=[N-]',
    'C12C(NC(N1)=O)CSC2',
    'c1c([OH])c([OH,NH2,NH])ccc1',
    'P',
    '[N,O,S]C#N',
    'C=C=O',
    '[Si][F,Cl,Br,I]',
    '[SX2]O',
    '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
    'O1CCCCC1OC2CCC3CCCCC3C2',
    'N=[CR0][N,n,O,S]',
    '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
    'C=[C!r]C#N',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
    '[OH]c1ccc([OH,NH2,NH])cc1',
    'c1ccccc1OC(=O)O',
    '[SX2H0][N]',
    'c12ccccc1(SC(S)=N2)',
    'c12ccccc1(SC(=S)N2)',
    'c1nnnn1C=O',
    's1c(S)nnc1NC=O',
    'S1C=CSC1=S',
    'C(=O)Onnn',
    'OS(=O)(=O)C(F)(F)F',
    'N#CC[OH]',
    'N#CC(=O)',
    'S(=O)(=O)C#N',
    'N[CH2]C#N',
    'C1(=O)NCC1',
    'S(=O)(=O)[O-,OH]',
    'NC[F,Cl,Br,I]',
    'C=[C!r]O',
    '[NX2+0]=[O+0]',
    '[OR0,NR0][OR0,NR0]',
    'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
    '[CX2R0][NX3R0]',
    'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
    '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
    '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
    '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
    '[*]=[N+]=[*]',
    '[SX3](=O)[O-,OH]',
    'N#N',
    'F.F.F.F',
    '[R0;D2][R0;D2][R0;D2][R0;D2]',
    '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
    'C=!@CC=[O,S]',
    '[#6,#8,#16][C,c](=O)O[C,c]',
    'c[C;R0](=[O,S])[C,c]',
    'c[SX2][C;!R]',
    'C=C=C',
    'c1nc([F,Cl,Br,I,S])ncc1',
    'c1ncnc([F,Cl,Br,I,S])c1',
    'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
    '[C,c]S(=O)(=O)c1ccc(cc1)F',
    '[15N]',
    '[13C]',
    '[18O]',
    '[34S]'
]

StructuralAlerts = []
for smarts in StructuralAlertSmarts:
    StructuralAlerts.append(Chem.MolFromSmarts(smarts))


# ADS parameters for the 8 molecular properties: [row][column]
#     rows[8]:     MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
#     columns[7]: A, B, C, D, E, F, DMAX
# ALOGP parameters from Gregory Gerebtzoff (2012, Roche)
pads1 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [0.486849448, 186.2293718, 2.066177165, 3.902720615,
             1.027025453, 0.913012565, 145.4314800],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
             0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
             12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
             1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
             1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]
# ALOGP parameters from the original publication
pads2 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [3.172690585, 137.8624751, 2.534937431, 4.581497897,
             0.822739154, 0.576295591, 131.3186604],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
             0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
             12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
             1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
             1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]

#====== druglikeliness


def ads(x, a, b, c, d, e, f, dmax):
    return ((a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f))))) / dmax)


def properties(mol):
    """
    Calculates the properties that are required to calculate the QED descriptor.
    """
    try:
        matches = []
        if mol is None:
            return None
        x = [0] * 9
        # MW
        x[0] = Descriptors.MolWt(mol)
        # ALOGP
        x[1] = Descriptors.MolLogP(mol)
        for hba in Acceptors:                                                        # HBA
            if mol.HasSubstructMatch(hba):
                matches = mol.GetSubstructMatches(hba)
                x[2] += len(matches)
        x[3] = Descriptors.NumHDonors(mol)                                            # HBD
        # PSA
        x[4] = Descriptors.TPSA(mol)
        x[5] = Descriptors.NumRotatableBonds(mol)                                    # ROTB
        x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(deepcopy(mol), AliphaticRings))    # AROM
        for alert in StructuralAlerts:                                                # ALERTS
            if (mol.HasSubstructMatch(alert)):
                x[7] += 1
        ro5_failed = 0
        if x[3] > 5:
            ro5_failed += 1  # HBD
        if x[2] > 10:
            ro5_failed += 1  # HBA
        if x[0] >= 500:
            ro5_failed += 1
        if x[1] > 5:
            ro5_failed += 1
        x[8] = ro5_failed
        return x
    except Exception as e:
        print(f"计算分子属性时出错: {str(e)}")
        return None


def qed_eval(w, p, gerebtzoff):
    try:
        if p is None:
            return 0.0
        d = [0.00] * 8
        if gerebtzoff:
            for i in range(0, 8):
                try:
                    d[i] = max(0.00001, ads(p[i], pads1[i][0], pads1[i][1], pads1[i][2], 
                              pads1[i][3], pads1[i][4], pads1[i][5], pads1[i][6]))
                except:
                    d[i] = 0.00001
        else:
            for i in range(0, 8):
                try:
                    d[i] = max(0.00001, ads(p[i], pads2[i][0], pads2[i][1], pads2[i][2],
                              pads2[i][3], pads2[i][4], pads2[i][5], pads2[i][6]))
                except:
                    d[i] = 0.00001
        
        t = 0.0
        for i in range(0, 8):
            t += w[i] * log(d[i])
        return exp(t / sum(w))
    except Exception as e:
        print(f"计算QED评分时出错: {str(e)}")
        return 0.0


def qed(mol):
    """
    Calculates the QED descriptor using average descriptor weights.
    If props is specified we skip the calculation step and use the props-list of properties.
    """
    try:
        if mol is None:
            return 0.0
        props = properties(mol)
        if props is None:
            return 0.0
        return qed_eval([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], props, True)
    except Exception as e:
        print(f"计算QED时出错: {str(e)}")
        return 0.0


def druglikeliness(smile, train_smiles):
    try:
        mol = Chem.MolFromSmiles(smile, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                            Chem.SANITIZE_SETAROMATICITY|
                            Chem.SANITIZE_SETCONJUGATION|
                            Chem.SANITIZE_SETHYBRIDIZATION|
                            Chem.SANITIZE_SYMMRINGS,
                        catchErrors=True)
                val = qed(mol)
                return val
            except:
                return 0.0
        return 0.0
    except:
        return 0.0


if __name__ == "__main__":
    import os
    # 创建一个列表存储所有批次的统计信息
    all_stats = []
    
    # 基础路径
    # base_path = 'C:\\Users\\23163\\Desktop\\NA_GAN\\RESULTS\\ACseqGAN\\NAPro\\final\\CRC\\epoch_data'
    base_path = '/ihome/jwang/hat170/ACORGAN/epoch_data/'
    save_path = '/ihome/jwang/hat170/ACORGAN/results/'
    if not os.path.exists(base_path):
        print(f"错误：基础路径不存在: {base_path}")
        exit(1)
    
    # 循环处理
    processed_files = 0
    for j in range(0, 2):
        for i in range(0, 50):
            # 读取对应批次的文件
            # input_file = os.path.join(base_path, f'try_classifier_{i}_{j}.csv')
            input_file = os.path.join(base_path, f'NAPro_718_{i}_{j}.csv')
            
            # 检查文件是否存在
            if not os.path.exists(input_file):
                print(f"跳过：文件不存在: {input_file}")
                continue
                
            print(f"\n正在处理文件: {input_file}")
            try:
                # 读取CSV文件
                df = pd.read_csv(input_file)
                processed_files += 1
                
                # 检查是否包含smiles列
                if 'smiles' not in df.columns:
                    print(f"错误：文件 {input_file} 中没有找到'smiles'列")
                    continue
                
                # 确保文件不为空
                if df.empty:
                    print(f"警告: 文件 {input_file} 为空")
                    continue
                
                print(f"发现 {len(df)} 个分子")
                
                # 计算每个分子的QED值
                qed_values = []
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
                                qed_val = qed(mol)
                                qed_values.append(qed_val)
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
                print(f"- 成功计算QED: {len(qed_values)}")
                
                # 计算统计信息
                if qed_values:
                    avg_qed = np.mean(qed_values)
                    std_qed = np.std(qed_values)
                    total_molecules = len(df)
                    valid_qed_count = len(qed_values)
                    
                    # 收集统计信息
                    stats = {
                        'batch_id': i,
                        'classifier_id': j,
                        'total_molecules': total_molecules,
                        'valid_qed_count': valid_qed_count,
                        'invalid_smiles': invalid_smiles,
                        'invalid_mols': invalid_mols,
                        'sanitize_errors': sanitize_errors,
                        'average_qed': round(avg_qed, 4),
                        'std_qed': round(std_qed, 4)
                    }
                    all_stats.append(stats)
                    
                    # 打印当前批次统计信息
                    print(f"\n批次 {i}_{j} 统计信息:")
                    print(f"总分子数: {total_molecules}")
                    print(f"有效QED计算数: {valid_qed_count}")
                    print(f"平均QED值: {avg_qed:.4f}")
                    print(f"QED标准差: {std_qed:.4f}")
                    
                    # 将QED值添加到数据框并保存
                    # df['qed'] = pd.Series(qed_values + [None] * (len(df) - len(qed_values)))
                    # output_file = os.path.join(base_path, f'try_classifier_{i}_{j}_qed.csv')
                    # df.to_csv(output_file, index=False)
                    # print(f"结果已保存至: {output_file}")
                
            except Exception as e:
                print(f"处理批次 {i}_{j} 时出错: {str(e)}")
                continue
    
    print(f"\n总共处理了 {processed_files} 个文件")
    
    # 将所有统计信息保存到汇总文件
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_output_file = os.path.join(save_path, 'qed_statistics_summary_718.csv')
        stats_df.to_csv(stats_output_file, index=False)
        print(f"\nQED统计汇总已保存至: {stats_output_file}")
    else:
        print("\n警告：没有成功处理任何批次的数据")

