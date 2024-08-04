import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import dvc.api

def load_data():
    """
    Load dataset from DVC storage.
    """
    data_url = dvc.api.get_url(path='data/winequality-red.csv', repo='.')
    df = pd.read_csv(data_url)
    X = df.drop(columns=['quality'])
    y = (df['quality'] > 6).astype(int)  # Binary classification: quality > 6
    return X, y

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    
    # Load the dataset
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = LogisticRegression(max_iter=200)

    # Define the parameter grid
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'solver': ['liblinear', 'lbfgs']}

    # Perform hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Record the best model and parameters with MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(best_model, "model")

        # Log dataset version
        dataset_version = dvc.api.get_url(path='data/winequality-red.csv', repo='.')
        mlflow.log_param("dataset_version", dataset_version)

        print(f"Best model with params={best_params} ended with accuracy={accuracy}, f1={f1}, and dataset_version={dataset_version}")
