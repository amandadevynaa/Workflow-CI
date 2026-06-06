import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import warnings
import argparse
import os

warnings.filterwarnings("ignore")

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='iris_preprocessing.csv')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# SETUP MLflow: THE ULTIMATE OVERRIDE
tracking_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{tracking_path}")
# Baris set_experiment DIHAPUS agar MLflow otomatis masuk ke default experiment (mlruns/0)
# yang dibutuhkan oleh Dockerfile di GitHub Actions.

# LOAD PREPROCESSED DATA
df = pd.read_csv(args.data_path)

# SPLIT FEATURES AND TARGET
X = df.drop(columns=['Species'])
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

# APPLY SMOTE
smote = SMOTE(random_state=args.random_state)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# TRACKING & TRAINING
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="RandomForest_Iris_Baseline") as run:
    # Model definition
    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)

    # EVALUATION
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics explicitly
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the model explicitly
    mlflow.sklearn.log_model(model, "model")

    # FORMATTED OUTPUT
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    print("="*50 + "\n")

    # Print run_id for easy extraction
    print(f"MLflow Run ID: {run.info.run_id}")
    
    # Save run_id to a file for easy retrieval in CI/CD pipelines
    with open("mlflow_run_id.txt", "w") as f:
        f.write(run.info.run_id)