# pylint: disable=<C0103, C0301, W0621>
"""Python script to do batch prediction using a pre-trained model."""

import os
import pickle

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from prefect import flow, task
from sqlalchemy import create_engine, text

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


def load_db_credentials_from_env(env_path=".env"):
    """Load database credentials from a .env file."""
    load_dotenv(env_path)
    return {
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def get_sqlalchemy_engine(db_creds):
    """Create a SQLAlchemy engine for PostgreSQL."""
    url = f"postgresql+psycopg2://{db_creds['user']}:{db_creds['password']}@database/batch"
    return create_engine(url)


def save_to_postgres(df, db_creds, table_name="stroke_predictions"):
    """Save the DataFrame to a PostgreSQL database."""
    engine = get_sqlalchemy_engine(db_creds)
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        input_id VARCHAR(64),
        gender INT,
        age FLOAT,
        hypertension INT,
        heart_disease INT,
        ever_married INT,
        work_type INT,
        Residence_type INT,
        avg_glucose_level FLOAT,
        bmi FLOAT,
        smoking_status INT,
        stroke INT,
        stroke_pred INT,
        model_id VARCHAR(64)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        for _, row in df.iterrows():
            conn.execute(
                text(
                    f"""
                    INSERT INTO {table_name} (
                        input_id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                        avg_glucose_level, bmi, smoking_status, stroke, stroke_pred, model_id
                    ) VALUES (
                        :input_id, :gender, :age, :hypertension, :heart_disease, :ever_married, :work_type, :Residence_type,
                        :avg_glucose_level, :bmi, :smoking_status, :stroke, :stroke_pred, :model_id
                    )
                """
                ),
                {
                    "input_id": str(row.get("id", "")),
                    "gender": int(row["gender"]),
                    "age": float(row["age"]),
                    "hypertension": int(row["hypertension"]),
                    "heart_disease": int(row["heart_disease"]),
                    "ever_married": int(row["ever_married"]),
                    "work_type": int(row["work_type"]),
                    "Residence_type": int(row["Residence_type"]),
                    "avg_glucose_level": float(row["avg_glucose_level"]),
                    "bmi": float(row["bmi"]),
                    "smoking_status": int(row["smoking_status"]),
                    "stroke": int(row.get("stroke", -1)),
                    "stroke_pred": int(row["stroke_pred"]),
                    "model_id": str(row["model_id"]),
                },
            )


@flow
def run_batch(
    input_file: str,
    env_path: str = ".env",
    model_dir: str = "./models",
    table_name: str = "stroke_predictions",
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

    print("...loading DB credentials from .env")
    db_creds = load_db_credentials_from_env(env_path)

    print("...saving to postgres")
    save_to_postgres(df_result, db_creds, table_name=table_name)

    print("...done")


if __name__ == "__main__":
    run_batch.serve(
        name="batch-predict",
        parameters={
            "input_file": "./data/stroke-data-2.csv",
            "env_path": ".env",
            "model_dir": "./models",
            "table_name": "stroke_predictions_2",
        },
    )
