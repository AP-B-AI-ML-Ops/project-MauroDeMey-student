# pylint: disable=<C0103>
"""Register the best Random Forest model from Hyperopt runs."""
import os
import pickle

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_pickle(filename: str):
    """Load an object from a pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_and_log_model(params: dict):
    """Train a Random Forest model and log it to MLflow."""
    mlflow.set_tracking_uri("http://experiment-tracking:5000")
    mlflow.set_experiment("random-forest-best-models")
    # Load preprocessed training and validation data
    X_train, y_train = load_pickle(os.path.join("./models", "train.pkl"))
    X_test, y_test = load_pickle(os.path.join("./models", "test.pkl"))

    with mlflow.start_run():
        for param in [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "n_jobs",
        ]:
            params[param] = int(params[param])

        # Train the Random Forest Classifier
        rf_classifier = RandomForestClassifier(**params)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(rf_classifier, artifact_path="model")


@flow
def run_register_model(top_n: int):
    """Register the best Random Forest model from Hyperopt runs."""
    mlflow.set_tracking_uri("http://experiment-tracking:5000")
    mlflow.set_experiment("random-forest-best-models")

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name("random-forest-hyperopt")
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"],
    )
    for run in runs:
        raw_params = dict(run.data.params)
        # Validate parameter schema
        validated_params = {
            "n_estimators": int(raw_params.get("n_estimators", 100)),
            "max_depth": int(raw_params.get("max_depth", 10)),
            "min_samples_split": int(raw_params.get("min_samples_split", 2)),
            "min_samples_leaf": int(raw_params.get("min_samples_leaf", 1)),
            "random_state": int(raw_params.get("random_state", 42)),
            "n_jobs": int(raw_params.get("n_jobs", -1)),
        }
        train_and_log_model(params=validated_params)

    # Select the model with the highest test accuracy
    experiment = client.get_experiment_by_name("random-forest-best-models")
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"],
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")


if __name__ == "__main__":
    run_register_model.serve(name="register-model-flow", parameters={"top_n": 5})
