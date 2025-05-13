import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split',
             'min_samples_leaf', 'random_state', 'n_jobs']

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename: str):
    """Load an object from a pickle file."""
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


def train_and_log_model(params):
    # Load preprocessed training and validation data
    X_train, y_train = load_pickle(os.path.join('./models', 'train.pkl'))
    X_test, y_test = load_pickle(os.path.join('./models', 'test.pkl'))

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        # Train the Random Forest Classifier
        rf_classifier = RandomForestClassifier(**params)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        # Log the model
        mlflow.sklearn.log_model(rf_classifier, artifact_path="model")


def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"]
    )
    for run in runs:
        train_and_log_model(params=run.data.params)

    # Select the model with the highest test accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")


if __name__ == "__main__":
    print("Registering model...")
    run_register_model("./models/", 5)
