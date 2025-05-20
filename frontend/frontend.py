# pylint: disable=<C0103>
"""Flask app for stroke prediction using a machine learning model."""
import pickle

import mlflow
import pandas as pd
from flask import Flask, render_template_string, request
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient("http://experiment-tracking:5000")

app = Flask("prediction")

model = None


def load_latest_best_model():
    """Load the latest best model from MLflow."""
    # Replace 'best-model' with your registered model name in MLflow
    model_name = "rf-best-model"
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    # Sort by version number (as int), descending
    latest_version = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest_version.version}"
    return mlflow.pyfunc.load_model(model_uri)


categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
numerical_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
]


def predict(features):
    """Predict the stroke probability using the loaded model."""
    predictions = model.predict(features)
    return float(predictions[0])


def prepare_features(stroke):
    """Prepare the features for prediction."""
    df = pd.DataFrame([stroke])
    for col in categorical_columns:
        with open(f"./models/{col}_label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        df[col] = label_encoder.transform(df[col])
    with open("./models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df


@app.before_request
def reload_model():
    """Reload the model before each request."""
    global model  # pylint: disable=<W0603>
    model = load_latest_best_model()


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the index page and handle form submission."""
    result = "None"
    if request.method == "POST":
        form = request.form
        # Map Yes/No to 1/0 for hypertension and heart_disease
        stroke = {
            "gender": form["gender"],
            "age": float(form["age"]),
            "hypertension": 1 if form["hypertension"] == "Yes" else 0,
            "heart_disease": 1 if form["heart_disease"] == "Yes" else 0,
            "ever_married": form["ever_married"],
            "work_type": form["work_type"],
            "Residence_type": form["Residence_type"],
            "avg_glucose_level": float(form["avg_glucose_level"]),
            "bmi": float(form["bmi"]),
            "smoking_status": form["smoking_status"],
        }
        features = prepare_features(stroke)
        prediction = predict(features)
        # Display "Stroke" if prediction == 1, else "No stroke"
        if round(prediction) == 1:
            result = "Stroke"
        elif round(prediction) == 0:
            result = "No stroke"
        else:
            result = "None"

    html = """
    <h2>Stroke Prediction</h2>
    <form method="post">
      Gender:
      <select name="gender">
        <option value="Male">Male</option>
        <option value="Female">Female</option>
      </select><br>
      Age: <input name="age" type="number" step="any"><br>
      Hypertension:
      <select name="hypertension">
        <option value="No">No</option>
        <option value="Yes">Yes</option>
      </select><br>
      Heart Disease:
      <select name="heart_disease">
        <option value="No">No</option>
        <option value="Yes">Yes</option>
      </select><br>
      Ever Married:
      <select name="ever_married">
        <option value="No">No</option>
        <option value="Yes">Yes</option>
      </select><br>
      Work Type:
      <select name="work_type">
        <option value="Private">Private</option>
        <option value="Govt_job">Govt_job</option>
        <option value="children">children</option>
        <option value="Self-employed">Self-employed</option>
        <option value="Never_worked">Never_worked</option>
      </select><br>
      Residence Type:
      <select name="Residence_type">
        <option value="Rural">Rural</option>
        <option value="Urban">Urban</option>
      </select><br>
      Avg Glucose Level: <input name="avg_glucose_level" type="number" step="any"><br>
      BMI: <input name="bmi" type="number" step="any"><br>
      Smoking Status:
      <select name="smoking_status">
        <option value="smokes">smokes</option>
        <option value="formerly smoked">formerly smoked</option>
        <option value="never smoked">never smoked</option>
        <option value="Unknown">Unknown</option>
      </select><br>
      <input type="submit" value="Predict">
    </form>
    {% if result %}
      <h3>{{ result }}</h3>
    {% endif %}
    """
    return render_template_string(html, result=result)


@app.route("/predict", methods=["POST"])
def predict_endpoint_post():
    """Predict the stroke probability using a POST request."""
    stroke = request.get_json()
    features = prepare_features(stroke)
    prediction = predict(features)
    return {"tip amount": prediction}


if __name__ == "__main__":
    app.run(port=9696, host="0.0.0.0", debug=True)
