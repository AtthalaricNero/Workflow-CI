import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DAGSHUB_USERNAME = "AtthalaricNero"
DAGSHUB_REPO_NAME = "Proyek_ML_Dicoding"
remote_server_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

mlflow.set_tracking_uri(remote_server_uri)

def main():
    print("--- Memulai Proses Training CI/CD ---")
    
    csv_filename = 'clean_telco_churn_preprocessing.csv' 
    
    try:
        df = pd.read_csv(csv_filename)
        print(f"Berhasil memuat data: {csv_filename}")
    except FileNotFoundError:
        print(f"Error Fatal: File '{csv_filename}' tidak ditemukan di folder kerja saat ini.")
        print("Isi folder saat ini:", os.listdir())
        return

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Sedang mencari parameter terbaik...")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    y_pred = best_model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    mlflow.sklearn.autolog(disable=True) 

    with mlflow.start_run() as run:
        print("Logging ke MLflow...")
        
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model_final")
        
        # Artefak
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10,6))
            pd.Series(best_model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        
        print(f"Run ID ({run_id}) berhasil disimpan ke 'run_id.txt'")
        print("Pipeline Selesai.")

if __name__ == "__main__":
    main()