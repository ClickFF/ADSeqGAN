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

# Function to load data
def classifier_data_loader(filepath):
    data = pd.read_csv(filepath)
    return data

# Function to calculate molecular descriptors
def calculate_descriptors(smiles_list, descriptor_names:list=None):
    if descriptor_names is None:
        descriptor_names = [
            'NumAromaticRings', 'HallKierAlpha', 'BertzCT', 'PEOE_VSA8', 'VSA_EState6', 'NumAromaticCarbocycles',
            'SlogP_VSA6', 'SMR_VSA7', 'MolMR', 'BalabanJ', 'fr_bicyclic', 'MinEStateIndex',
            'Chi1', 'FpDensityMorgan1', 'Chi1n', 'Chi0n', 'LabuteASA', 'Ipc'
            # 'TPSA', 'NumHDonors', 'NOcount', 'NumHeteroatoms', 'NumHAcceptors', 'VSA_EState3', 'SMR_VSA1', 'MinEStateIndex', 
            # 'PEOE_VSA1', 'Kappa3', 'ExactMolWt', 'Chi0' 
            # 'FractionCSP3', 'SMR_VSA5', 'fr_aniline', 'NumAromaticHeterocycles', 'SPS', 'HallKierAlpha', 'SlogP_VSA10',
            # 'NumHAcceptors', 'TPSA', 'VSA_EState8', 'PEOE_VSA6', 'Chi4n', 'NumSaturatedRings', 'Chi3n', 'Chi4v'
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
        model_path = os.path.join(current_dir, 'molecular_classifier.pkl')
    
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

# Example usage
if __name__ == "__main__":
    prior_classifier('train_CNS.csv', from_file=True)
    # Single molecule prediction
    # test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    # result = predict_molecule(test_smiles)
    # if result['success']:
    #     print(f"Prediction: {result['prediction']}")
    #     print(f"Probability: {result['probability']:.3f}")
    # else:
    #     print(f"Error: {result['error']}")
    
    # # Batch prediction
    # test_smiles_list = [
    #     "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    #     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    # ]
    # results = batch_predict(test_smiles_list)
    # for smiles, result in zip(test_smiles_list, results):
    #     if result['success']:
    #         print(f"\nMolecule: {smiles}")
    #         print(f"Prediction: {result['prediction']}")
    #         print(f"Probability: {result['probability']:.3f}")
    #     else:
    #         print(f"\nMolecule: {smiles}")
    #         print(f"Error: {result['error']}")