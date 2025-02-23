import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os
import multiprocessing as mp
from itertools import product

# Function to load data
def classifier_data_loader(filepath):
    data = pd.read_csv(filepath)
    return data

# Function to calculate molecular descriptors
def calculate_descriptors(smiles_list, descriptor_names:list=None):
    if descriptor_names is None:
        descriptor_names = [
            # 'NumAromaticRings', 'HallKierAlpha', 'BertzCT', 'PEOE_VSA8', 'VSA_EState6', 'NumAromaticCarbocycles',
            # 'SlogP_VSA6', 'SMR_VSA7', 'MolMR', 'BalabanJ', 'fr_bicyclic', 'MinEStateIndex',
            # 'Chi1', 'FpDensityMorgan1', 'Chi1n', 'Chi0n', 'LabuteASA', 'Ipc'
            'TPSA', 'NumHDonors', 'NOcount', 'NumHeteroatoms', 'NumHAcceptors', 'VSA_EState3', 'SMR_VSA1', 'MinEStateIndex', 
            'PEOE_VSA1', 'Kappa3', 'ExactMolWt', 'Chi0' 
        ]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                            Chem.SANITIZE_SETAROMATICITY|
                            Chem.SANITIZE_SETCONJUGATION|
                            Chem.SANITIZE_SETHYBRIDIZATION|
                            Chem.SANITIZE_SYMMRINGS,
                        catchErrors=True)
                descriptors.append(calculator.CalcDescriptors(mol))
            except:
                descriptors.append([np.nan] * len(descriptor_names))
        else:
            descriptors.append([np.nan] * len(descriptor_names))
            
    return pd.DataFrame(descriptors, columns=descriptor_names)

# Function to train the model
def model_training(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    auc_scores = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train_fold, y_train_fold)
        if len(np.unique(y_train_fold)) > 1:  # Ensure there are at least two classes
            y_proba = clf.predict_proba(X_test_fold)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_fold, y_proba, pos_label=clf.classes_[1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            auc_scores.append(auc(fpr, tpr))
    
    return clf, tprs, mean_fpr

# Function to output the ROC curve figure
def output_figure(tprs, mean_fpr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')

    for i, tpr in enumerate(tprs):
        plt.plot(mean_fpr, tpr, linestyle='--', alpha=0.3)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest - Five Fold Cross Validation')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))

# Function to train and evaluate the classifier
def prior_classifier(data, from_file=False):
    """Train and evaluate the classifier
    
    Args:
        data: Either a file path (if from_file=True) or a list of [smiles, label] pairs
        from_file: Boolean indicating whether data is a file path
    """
    # Load and prepare data
    if from_file:
        # Load data from file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', data))
        data = classifier_data_loader(data_path)
        smiles_list = data['smiles'].tolist()
        labels = data['label'].tolist()
    
    # Calculate molecular descriptors
    else:
        smiles_list, labels = zip(*data)
    descriptor_df = calculate_descriptors(smiles_list)
    descriptor_df['label'] = labels
    descriptor_df = descriptor_df.dropna()
        
    X = descriptor_df.drop('label', axis=1)
    y = descriptor_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and evaluate
    clf, tprs, mean_fpr = model_training(X, y)
    
    # Output figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'eval_classifier')
    output_figure(tprs, mean_fpr, output_dir)
    
    # Train final model and save it
    clf.fit(X_train, y_train)
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(clf, X.columns, X, y)
    
    model_path = os.path.join(current_dir, 'molecular_classifier.pkl')
    joblib.dump(clf, model_path)

def load_model(model_path=None):
    """Load the trained molecular classifier model
    
    Args:
        model_path (str, optional): Path to the model file. If None, will try to load from default location.
    
    Returns:
        The loaded model
    """
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'molecular_classifier_cns.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    return joblib.load(model_path)

def predict_molecule(smiles, model=None, threshold=0.5):
    """Predict whether a molecule is active using the trained model
    
    Args:
        smiles (str): SMILES string of the molecule
        model: Pre-loaded model (optional). If None, will load the model from default location
        threshold (float): Probability threshold for binary classification
    
    Returns:
        dict: Dictionary containing prediction results:
            - 'prediction': Binary prediction (0 or 1)
            - 'probability': Probability of being active
            - 'success': Whether prediction was successful
            - 'error': Error message if prediction failed
    """
    try:
        # Calculate molecular descriptors
        descriptors = calculate_descriptors([smiles])
        if descriptors.isnull().values.any():
            return {
                'success': False,
                'error': 'Invalid SMILES string or failed to calculate descriptors'
            }
        
        # Load or use provided model
        if model is None:
            model = load_model()
        
        # Make prediction
        prob = model.predict_proba(descriptors)[0][1]
        pred = 1 if prob >= threshold else 0
        
        return {
            'success': True,
            'prediction': pred,
            'probability': prob,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def batch_predict(smiles_list, model=None, threshold=0.5):
    """Predict multiple molecules at once
    
    Args:
        smiles_list (list): List of SMILES strings
        model: Pre-loaded model (optional). If None, will load the model from default location
        threshold (float): Probability threshold for binary classification
    
    Returns:
        list: List of prediction results for each molecule
    """
    # Load model once for batch prediction
    if model is None:
        model = load_model()
    
    return [predict_molecule(smiles, model, threshold) for smiles in smiles_list]

# 在model_training函数后添加新的函数
def analyze_feature_importance(clf, feature_names, X, y):
    """Analyze feature importance and class distributions
    
    Args:
        clf: Trained random forest classifier
        feature_names: List of feature names
        X: Feature matrix
        y: Target labels
    """
    # Get feature importance
    importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Get top 5 features
    top_5_features = feature_importance_df['feature'].head().tolist()
    
    # Analyze class distributions for top features
    print("\nClass Distribution Analysis for Top 5 Features:")
    for feature in top_5_features:
        # Get feature values for each class
        class_0_values = X[feature][y == 0]
        class_1_values = X[feature][y == 1]
        
        # Calculate statistics
        stats = {
            'Class 0': {
                'mean': np.mean(class_0_values),
                'median': np.median(class_0_values),
                'std': np.std(class_0_values),
                'min': np.min(class_0_values),
                'max': np.max(class_0_values)
            },
            'Class 1': {
                'mean': np.mean(class_1_values),
                'median': np.median(class_1_values),
                'std': np.std(class_1_values),
                'min': np.min(class_1_values),
                'max': np.max(class_1_values)
            }
        }
        
        print(f"\nFeature: {feature}")
        print(f"Importance Score: {feature_importance_df[feature_importance_df['feature'] == feature]['importance'].values[0]:.4f}")
        print("\nClass 0 Statistics:")
        print(f"Mean ± Std: {stats['Class 0']['mean']:.4f} ± {stats['Class 0']['std']:.4f}")
        print(f"Median: {stats['Class 0']['median']:.4f}")
        print(f"Range: [{stats['Class 0']['min']:.4f}, {stats['Class 0']['max']:.4f}]")
        
        print("\nClass 1 Statistics:")
        print(f"Mean ± Std: {stats['Class 1']['mean']:.4f} ± {stats['Class 1']['std']:.4f}")
        print(f"Median: {stats['Class 1']['median']:.4f}")
        print(f"Range: [{stats['Class 1']['min']:.4f}, {stats['Class 1']['max']:.4f}]")
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(class_0_values, bins=30, alpha=0.5, label='Class 0', color='blue')
        plt.hist(class_1_values, bins=30, alpha=0.5, label='Class 1', color='red')
        plt.axvline(stats['Class 0']['mean'], color='blue', linestyle='dashed', alpha=0.8)
        plt.axvline(stats['Class 1']['mean'], color='red', linestyle='dashed', alpha=0.8)
        plt.title(f'{feature} Distribution by Class')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        # Save distribution plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, '..', 'eval_classifier')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'class_dist_{feature}.pdf'))
        plt.close()
    
    # Original feature importance plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), feature_importance_df['importance'])
    plt.xticks(range(len(importances)), feature_importance_df['feature'], rotation=45, ha='right')
    plt.xlabel('Descriptors')
    plt.ylabel('Feature Importance')
    plt.title('Random Forest Classifier - Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'))
    plt.close()
    
    return feature_importance_df

def process_batch(args):
    """处理单个批次的函数"""
    i, j = args
    try:
        input_file = f'/ihome/jwang/hat170/ACORGAN/epoch_data/CNS_1234_{i}_{j}.csv'
        df = pd.read_csv(input_file)
        
        if df.empty:
            print(f"警告：文件 {input_file} 为空")
            return None
            
        model = load_model()
        print(f"模型已加载，正在处理批次 {i}_{j}")
        results = batch_predict(df['smiles'].tolist(), model=model)
        
        df['predict_class'] = [result['prediction'] if result['success'] else -1 for result in results]
        df['predict_probability'] = [result['probability'] if result['success'] else -1 for result in results]
        
        total_molecules = len(df)
        successful_predictions = sum(1 for result in results if result['success'])
        failed_predictions = total_molecules - successful_predictions
        unique_smiles = df['smiles'].nunique()
        unique_ratio = unique_smiles / total_molecules if total_molecules > 0 else 0
        
        valid_smiles = [smiles for smiles, result in zip(df['smiles'], results) if result['success']]
        if valid_smiles:
            smiles_lengths = [len(smiles) for smiles in valid_smiles]
            avg_length = sum(smiles_lengths) / len(smiles_lengths)
            length_sd = np.sqrt(sum((x - avg_length) ** 2 for x in smiles_lengths) / len(smiles_lengths))
        else:
            avg_length = length_sd = 0
        
        return {
            'batch_id': i,
            'classifier_id': j,
            'total_molecules': total_molecules,
            'unique_molecules': unique_smiles,
            'unique_ratio': round(unique_ratio, 4),
            'avg_smiles_length': round(avg_length, 2),
            'length_sd': round(length_sd, 2),
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'class_0_count': df['predict_class'].value_counts().get(0, 0),
            'class_1_count': df['predict_class'].value_counts().get(1, 0)
        }
    except Exception as e:
        print(f"处理批次 {i}_{j} 时出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 创建所有批次的参数组合
    batch_params = list(product(range(50), range(2)))
    
    # 获取CPU核心数，留出1-2个核心给系统使用
    n_cores = 4
    print(f"使用 {n_cores} 个CPU核心进行并行处理")
    
    # 创建进程池并执行并行处理
    with mp.Pool(n_cores) as pool:
        all_stats = pool.map(process_batch, batch_params)
    
    # 过滤掉None结果并保存统计信息
    all_stats = [stat for stat in all_stats if stat is not None]
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_output_file = '/ihome/jwang/hat170/ACORGAN/results/CNS_1234_prediction_statistics_summary.csv'
        stats_df.to_csv(stats_output_file, index=False)
        print(f"\n统计汇总已保存至: {stats_output_file}")
    else:
        print("\n警告：没有成功处理任何批次的数据")