# pylint: disable=<C0103>
"""Train a Random Forest Classifier and log the model with MLflow."""
import os
import pickle

import mlflow
from prefect import flow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment("random-forest-train")


def load_pickle(filename: str):
    """Load an object from a pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@flow
def run_train(data_path: str):
    """Main function to run training."""
    mlflow.sklearn.autolog()
    # Load the preprocessed data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():

        # Initialize the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Train the model
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    run_train.serve(name="train-flow", parameters={"data_path": "./models"})
