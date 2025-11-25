import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.io import arff

# === Mapp där dina dataset ligger ===
DATA_PATH = r"C:\Users\josef\OneDrive\Desktop\Thesis\data"

# === Funktioner ===
def list_datasets():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.csv', '.arff'))]
    if not files:
        print("⚠️ Inga dataset hittades i mappen.")
        exit()
    return files

def load_dataset(filename):
    full_path = os.path.join(DATA_PATH, filename)
    if filename.endswith(".csv"):
        df = pd.read_csv(full_path)
    elif filename.endswith(".arff"):
        data, meta = arff.loadarff(full_path)
        df = pd.DataFrame(data)
        # konvertera byte-kolumner till strängar
        df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
    else:
        raise ValueError("Unsupported file type.")
    print(f"✅ Loaded {filename} with shape {df.shape}")
    return df

def preprocess_data(df):
    # försök hitta kolumnen för defekter
    target_candidates = [c for c in df.columns if 'defect' in c.lower() or 'bug' in c.lower()]
    if not target_candidates:
        raise ValueError("Ingen kolumn för 'defects' hittades!")
    target = target_candidates[0]
    
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    smote = SMOTE(k_neighbors=5, sampling_strategy=0.7, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test

def train_and_evaluate(model, X_res, X_test, y_res, y_test):
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    print("\n=== Result ===")
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))

# === Meny ===
datasets = list_datasets()
print("\n=== Choose Dataset ===")
for i, d in enumerate(datasets, 1):
    print(f"{i}. {d}")
dataset_choice = int(input("Enter number: ")) - 1
df = load_dataset(datasets[dataset_choice])

models = ["RandomForest", "XGBoost"]
print("\n=== Choose Model ===")
for i, m in enumerate(models, 1):
    print(f"{i}. {m}")
model_choice = int(input("Enter number: ")) - 1

X_res, X_test, y_res, y_test = preprocess_data(df)

# === Modelltuning ===
if models[model_choice] == "RandomForest":
    n_estimators = int(input("Number of trees (default 200): ") or 200)
    max_depth = int(input("Max depth (default 12): ") or 12)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

elif models[model_choice] == "XGBoost":
    n_estimators = int(input("Number of estimators (default 300): ") or 300)
    learning_rate = float(input("Learning rate (default 0.1): ") or 0.1)
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                          use_label_encoder=False, eval_metric='logloss')

print(f"\nTraining {models[model_choice]} on {datasets[dataset_choice]}...\n")
train_and_evaluate(model, X_res, X_test, y_res, y_test)
