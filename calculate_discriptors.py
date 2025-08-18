from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Descriptors3D
import pandas as pd

# =============================================================================
# INPUT AND OUTPUT PATHS - MODIFY THESE AS NEEDED
# =============================================================================

# Input file path
INPUT_CSV_PATH = '/PATH/TO/train.csv'
INPUT_ENCODING = 'gbk'

# Output file paths
OUTPUT_DESCRIPTORS_PATH = 'PATH/TO/molecular_descriptors_CNS.csv'
OUTPUT_AUC_PATH = 'PATH/TO/descriptor_auc_scores_CNS.csv'

# =============================================================================

def calculate_descriptors(smiles):
    """
    Caculate all the molecular descriptors of the given SMILES, including 3D descriptors.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None  :
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                          Chem.SANITIZE_SETAROMATICITY|
                          Chem.SANITIZE_SETCONJUGATION|
                          Chem.SANITIZE_SETHYBRIDIZATION|
                          Chem.SANITIZE_SYMMRINGS,
                    catchErrors=True)
        except Exception as e:
            print(f"Failed to clean the molecule: {str(e)}")
            return None

    # Try to generate 3D conformations
    try:
        mol_h = Chem.AddHs(mol)  # Add hydrogens
        AllChem.EmbedMolecule(mol_h, randomSeed=42)  # Generate 3D conformations
        AllChem.UFFOptimizeMolecule(mol_h)  # Optimize the conformations
    except Exception as e:
        mol_h = None  # Failed to generate 3D conformations

    descriptors = {}

    # Calculate standard descriptors
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except Exception as e:
            descriptors[name] = None

    # Calculate 3D descriptors (if 3D conformations are successfully generated)
    if mol_h:
        for name, func in Descriptors3D.descList:
            try:
                descriptors[name] = func(mol_h)
            except Exception as e:
                descriptors[name] = None
    else:
        for name, _ in Descriptors3D.descList:
            descriptors[name] = None

    return descriptors


# Read the data
data = pd.read_csv(INPUT_CSV_PATH, encoding=INPUT_ENCODING)
smiles_list = data['smiles']
label_list = data['label']

# Create a list to store all the molecular descriptors
all_descriptors = []

# Calculate the descriptors for each SMILES
for i, smiles in enumerate(smiles_list):
    print(f"Processing molecule {i+1}/{len(smiles_list)}")  # Add progress hint
    desc = calculate_descriptors(smiles)
    if desc:
        desc['SMILES'] = smiles  # Add SMILES column
        desc['label'] = label_list[i]
        all_descriptors.append(desc)
    else:
        print(f"Warning: Failed to process SMILES: {smiles}")

# Convert the results to a DataFrame
descriptors_df = pd.DataFrame(all_descriptors)

# Move the SMILES column to the front
cols = ['SMILES'] + ['label'] + [col for col in descriptors_df.columns if col != 'SMILES' and col != 'label']
descriptors_df = descriptors_df[cols]

# Save the results to a CSV file
descriptors_df.to_csv(OUTPUT_DESCRIPTORS_PATH, index=False)
print(f"Descriptors calculated and saved to: {OUTPUT_DESCRIPTORS_PATH}")

# Output some basic statistics
print(f"\nTotal number of molecules processed: {len(all_descriptors)}")
print(f"Number of calculated descriptors: {len(descriptors_df.columns)-1}")  # -1 because we need to subtract the SMILES column

# Use regression to calculate the AUC of the descriptors for 0 and 1 class molecules, for classification, output to csv file
# Add the following code at the end of the file

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Start calculating the AUC values of the descriptors...")

# Re-read the data
descriptors_df = pd.read_csv(OUTPUT_DESCRIPTORS_PATH, encoding=INPUT_ENCODING)

# Remove the SMILES and label columns, only keep the descriptor columns
X = descriptors_df.drop(['SMILES', 'label'], axis=1)
y = descriptors_df['label']

# Remove columns with missing values
X = X.dropna(axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a dictionary to store the AUC values
auc_scores = {}

# Calculate the AUC for each descriptor
for i, descriptor in enumerate(X.columns):
    try:
        # Use a single descriptor to train the logistic regression model
        X_single = X_scaled[:, i].reshape(-1, 1)
        model = LogisticRegression(random_state=42)
        model.fit(X_single, y)
        
        y_pred_proba = model.predict_proba(X_single)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        
        auc_scores[descriptor] = auc
    except Exception as e:
        print(f"Error calculating the AUC of the descriptor {descriptor}: {str(e)}")
        continue

# Convert the AUC values to a DataFrame
auc_df = pd.DataFrame.from_dict(auc_scores, orient='index', columns=['AUC'])
auc_df = auc_df.sort_values('AUC', ascending=False)  # Sort by AUC values in descending order

# Add a column to evaluate the discriminative ability of the descriptors
auc_df['AUC_diff_from_0.5'] = abs(auc_df['AUC'] - 0.5)

# Save the results to a CSV file
auc_df.to_csv(OUTPUT_AUC_PATH)

# Output some statistics
print(f"\nAUC values calculated and saved to: {OUTPUT_AUC_PATH}")
print("\nThe top 10 descriptors with the highest discriminative ability:")
print(auc_df.head(10))
print(f"\nTotal number of descriptors calculated: {len(auc_df)}")

