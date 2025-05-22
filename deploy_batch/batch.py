# pylint: disable=<C0103, C0301, W0621>
"""Python script to do batch prediction using a pre-trained model."""

import os
import pickle

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task

mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient("http://experiment-tracking:5000")
model_name = "rf-best-model"


def read_dataframe(filename: str):
    """Read a CSV file into a pandas DataFrame."""
    df = pd.read_csv(filename)
    # Optionally: drop rows with missing values or preprocess as needed
    df = df.dropna()
    return df


def load_pickle(filename: str):
    """Load a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


@task
def preprocess_batch(df, model_dir="./models"):
    """Preprocess the input DataFrame for prediction."""
    # Load scalers and encoders
    ss = load_pickle(os.path.join(model_dir, "scaler.pkl"))
    # Load label encoders for each categorical column
    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    label_encoders = {}
    for col in categorical_columns:
        le_path = os.path.join(model_dir, f"{col}_label_encoder.pkl")
        label_encoders[col] = load_pickle(le_path)
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # Scale numerical columns
    numerical_columns = ["age", "avg_glucose_level", "bmi"]
    df[numerical_columns] = ss.transform(df[numerical_columns])

    return df


@task
def prep_df(df, model_dir="./models"):
    """Preprocess the DataFrame for prediction and return the DataFrame with only feature columns."""
    df = preprocess_batch(df, model_dir=model_dir)
    feature_cols = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ]
    return df[feature_cols]


def get_latest_version(model_name):
    """Get the latest version of the model"""
    versions = client.get_latest_versions(model_name)
    # Use the latest version in "None" or "Production" stage
    if versions:
        return versions[-1].version
    raise RuntimeError("No model versions found.")


def load_model():
    """Load the latest version of the model from MLflow."""
    print("...loading model")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

    return model


def load_data(input_file):
    """Load the data from a CSV file."""
    print("...reading dataframe")
    df = read_dataframe(input_file)

    return df


@flow
def run_batch(
    input_file: str, output_dir: str = "./batch-output", model_dir: str = "./models"
):
    """Main function to run the batch prediction."""
    model = load_model()
    df = load_data(input_file)

    print("...prepping data")
    X = prep_df(df, model_dir=model_dir)

    print("...calculating prediction")
    y_pred = model.predict(X)

    print("...creating new dataframe")
    df_result = df.copy()
    df_result["stroke_pred"] = y_pred

    print(model.metadata.run_id)

    df_result["model_id"] = model.metadata.run_id

    print("...saving dataframe")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"stroke_predictions_{model.metadata.run_id}.csv"
    )
    print(f"...saving dataframe to: {output_path}")
    df_result.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_batch.serve(
        name="batch-predict",
        parameters={
            "input_file": "./data/stroke-data-2.csv",
            "output_dir": "./batch-output",
            "model_dir": "./models",
        },
    )
