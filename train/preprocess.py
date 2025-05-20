# pylint: disable=<C0103>
"""Preprocess the dataset for stroke prediction."""
import os
import pickle

import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def dump_pickle(obj, filename: str):
    """Dump an object to a pickle file."""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    """Read a DataFrame from a csv."""
    df = pd.read_csv(filename)
    return df


@task
def preprocess(df: pd.DataFrame, ss: StandardScaler, le: LabelEncoder, output_dir: str):
    """Preprocess the DataFrame."""
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Drop unnecessary columns
    df.drop(columns=["id"], inplace=True)

    # Encoding categorical features
    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
        dump_pickle(le, os.path.join(output_dir, f"{col}_label_encoder.pkl"))

    # Scaling numerical features
    numerical_columns = ["age", "avg_glucose_level", "bmi"]
    df[numerical_columns] = ss.fit_transform(df[numerical_columns])

    return df, ss, le


@flow
def run_data_prep(
    input_file: str, output_dir: str, test_size: float = 0.2, random_state: int = 42
):
    """Main function to run data preparation."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the dataset
    df = read_dataframe(input_file)

    # Initialize StandardScaler and LabelEncoder
    ss = StandardScaler()
    le = LabelEncoder()

    # Preprocess the DataFrame
    df, ss, le = preprocess(df, ss, le, output_dir)

    # Split the dataset into training and testing sets
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save the preprocessed data
    dump_pickle((X_train, y_train), os.path.join(output_dir, "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join(output_dir, "test.pkl"))


if __name__ == "__main__":
    run_data_prep.serve(
        name="data-prep-flow",
        parameters={
            "input_file": "./data/healthcare-dataset-stroke-data.csv",
            "output_dir": "./models",
            "test_size": 0.2,
            "random_state": 42,
        },
    )
