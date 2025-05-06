FROM python:3.13-slim

RUN pip install -U pip

RUN pip install psycopg2-binary mlflow

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql+psycopg2://postgres:postgres@database/mlflow_db"]

# --host omdat mlflow anders op 127.0.0.1 runt, en de port niet kan exposen naar buiten
# --port omdat we zeker willen zijn dat mlflow runt op port 5000 (belangrijk voor scripts)
# psycopg2 om mlflow de database te laten accessen
# mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri postgresql+psycopg2://postgres:postgres@database/mlflow_db