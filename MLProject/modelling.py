import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier

def main():
    X_train = pd.read_csv('dataset_preprocessing/X_train.csv')
    y_train = pd.read_csv('dataset_preprocessing/y_train.csv').values.ravel()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(model, "model")
        print("Model berhasil dilatih dan disimpan ke artefak MLflow lokal.")

if __name__ == '__main__':
    main()