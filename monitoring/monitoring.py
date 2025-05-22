"""Monitoring module for Evidently reports using Prefect"""

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import Flow, task
from sqlalchemy import create_engine, types
from sqlalchemy_utils import create_database, database_exists


@task
def load_env_vars():
    """Load environment variables from .env file"""
    load_dotenv()
    db_user = os.getenv("POSTGRES_USER")
    db_pwd = os.getenv("POSTGRES_PASSWORD")
    db_name = "batch"
    return db_user, db_pwd, db_name


@task
def get_engine(db_user, db_pwd, db_name):
    """Create a SQLAlchemy engine and create the database if it doesn't exist"""
    db_uri = f"postgresql+psycopg2://{db_user}:{db_pwd}@database/{db_name}"
    if not database_exists(db_uri):
        create_database(db_uri)
    engine = create_engine(db_uri)
    return engine


@task
def fetch_data(engine, table_name):
    """Fetch data from the specified table in the database"""
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


@task
def generate_report(reference_data, current_data):
    """Generate an Evidently report comparing reference and current data"""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    return snapshot


@task
def extract_metrics(snapshot):
    """Extract metrics from the Evidently report snapshot"""
    json_data = snapshot.dict()
    result_data = []
    report_time = datetime.now(timezone.utc)
    for metric in json_data["metrics"]:
        metric_id = metric["metric_id"]
        value = metric["value"]
        result_data.append(
            {"run_time": report_time, "metric_name": metric_id, "value": value}
        )
    return pd.DataFrame(result_data)


@task
def save_metrics(metrics_df, engine):
    """Save the metrics DataFrame to the database and generate an HTML report"""
    metrics_df.to_sql(
        "evidently_metrics",
        engine,
        if_exists="replace",
        index=False,
        dtype={"value": types.JSON},
    )
    metrics_df.to_html(
        "metrics/metrics.html",
        index=False,
        justify="center",
        border=0,
        classes="table table-striped table-bordered",
        table_id="metrics_table",
    )


@Flow
def main():
    """Main flow to run the Evidently monitoring process"""
    db_user, db_pwd, db_name = load_env_vars()
    engine = get_engine(db_user, db_pwd, db_name)
    reference_data = fetch_data(engine, "stroke_predictions_2")
    current_data = fetch_data(engine, "stroke_predictions_3")
    snapshot = generate_report(reference_data, current_data)
    metrics_df = extract_metrics(snapshot)
    save_metrics(metrics_df, engine)


if __name__ == "__main__":
    main.serve(
        name="Evidently monitoring Flow",
    )
