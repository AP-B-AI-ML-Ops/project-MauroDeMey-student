"""Unit tests for the preprocessing module."""

import tempfile

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train.preprocess import preprocess, read_dataframe


def test_read_dataframe(tmp_path):
    """Test the read_dataframe function."""
    # Create a sample CSV
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_path, index=False)
    df_loaded = read_dataframe(str(csv_path))
    assert df_loaded.equals(df)


def test_preprocess_basic():
    """Test the preprocess function with basic functionality."""
    df = pd.DataFrame(
        {
            "id": [1, 2],  # Add this line
            "gender": ["Male", "Female"],
            "age": [50, 60],
            "hypertension": [0, 1],
            "heart_disease": [0, 1],
            "ever_married": ["Yes", "No"],
            "work_type": ["Private", "Self-employed"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [100.0, 120.0],
            "bmi": [25.0, 30.0],
            "smoking_status": ["never smoked", "smokes"],
            "stroke": [0, 1],
        }
    )
    ss = StandardScaler()
    le = LabelEncoder()
    output_dir = tempfile.mkdtemp()
    df_out, _, _ = preprocess(df.copy(), ss, le, output_dir)
    assert "stroke" in df_out.columns
    assert not df_out.isnull().any().any()
