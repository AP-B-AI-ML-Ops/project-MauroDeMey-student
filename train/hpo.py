# pylint: disable=<C0103>
"""Hyperparameter optimization using Optuna and MLflow for Random Forest Classifier."""

import os
import pickle

import mlflow
import optuna
from prefect import flow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_pickle(filename: str):
    """Load an object from a pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@flow
def run_optimization(data_path: str, num_trials: int):
    """Main function to run hyperparameter optimization."""
    mlflow.set_tracking_uri("http://experiment-tracking:5000")
    mlflow.set_experiment("random-forest-hyperopt")
    # Load the preprocessed data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Disable autologging to avoid conflicts with Optuna
    mlflow.sklearn.autolog(disable=True)

    def objective(trial):
        # Define the hyperparameters to tune
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            # Create the model with the suggested hyperparameters
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

        return accuracy

    # Create a study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    print("Starting hyperparameter optimization...")
    run_optimization.serve(
        name="hpo-flow", parameters={"data_path": "./models", "num_trials": 10}
    )
