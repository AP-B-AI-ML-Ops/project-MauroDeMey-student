# MLOps-Project

**Made by Mauro De Mey, 2ITAI**

## Dataset(s)

I used the Stroke Prediction Dataset from Kaggle:  
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

This dataset contains health and demographic information to help predict the likelihood of a stroke. I split the dataset into training, and test sets using `sklearn`'s `train_test_split` function. To simulate new data for the service, I further divided the original dataset into separate subsets, allowing different parts of the data to be used as if they were new, unseen data.

## Project Explanation

The goal of this project is to demonstrate my understanding of MLOps concepts by building an end-to-end machine learning workflow. I set up automated training pipelines with logging using MLflow, orchestrated tasks with Prefect, and implemented monitoring with Evidently and Grafana, all running in Docker containers. The core application predicts the likelihood of a person having had or potentially having a stroke, based on their health and demographic information. This project showcases how to manage, monitor, and deploy machine learning models efficiently using modern MLOps tools.

## Flows & Actions

For this project, the following flows and actions are required to ensure a complete and robust MLOps pipeline:

### 1. Data Preprocessing, Training, Hyperparameter Optimization, and Model Registration Flow

- Load raw data from the dataset.
- Clean and preprocess the data (handle missing values, encode categorical variables, scale features, etc.).
- Split the data into training and test sets.
- Save the processed data for downstream tasks.
- Load the preprocessed training data.
- Train multiple models with different hyperparameters (using grid search or random search).
- Evaluate model performance on the validation set.
- Log metrics and parameters with MLflow.
- Select the best-performing model.
- Register the best model in the MLflow Model Registry.

### 2. Batch Prediction Flow

- Load the latest registered model.
- Load new, unseen batch data.
- Preprocess the batch data using the same steps as training.
- Generate predictions for the batch.
- Store predictions and relevant metadata for monitoring.

### 3. Data Drift Detection Flow

- Compare the distribution of features and predictions in the new batch data to the training data using Evidently.
- Generate drift reports and dashboards.
- Trigger alerts or logs if significant drift is detected.

These flows are orchestrated using Prefect, ensuring automation, reproducibility, and monitoring throughout the machine learning lifecycle.

## Set-up instructions

### Set-up Instructions

1. **Prerequisites**

   - Ensure you have [Docker](https://www.docker.com/products/docker-desktop/) installed and running.

2. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

3. **Environment Variables**

   - Create a `.env` file in each of the following folders: `database`, `deploy_batch`, and `monitoring`.
   - Use this template for each `.env` file:
     ```
     POSTGRES_USER=postgres
     POSTGRES_PASSWORD=postgres
     POSTGRES_DB=postgres
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     ```

4. **Dataset Preparation**

   - Download the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.
   - Place the dataset in `train/data/`.
   - In your terminal, navigate to `train/data` and run:
     ```bash
     python split_data.py
     ```
     This will split the data for later batch predictions.

5. **Start the Project**

   - From the root folder of the project, run:
     ```bash
     docker compose up
     ```
   - Wait for all containers to start.

6. **Accessing Services**

   - **Prefect UI:** [http://localhost:4200](http://localhost:4200)  
     Start by running the main flow.
   - **MLflow UI:** [http://localhost:5000](http://localhost:5000)  
     Check resulting models after running the train flow.
   - **Web App:** [http://localhost:9696](http://localhost:9696)  
     This web app won't work if you haven't run the trian flow yet, because you need to have a trained model in MLflow in order to use it.
     Fill in the form to get stroke predictions.
   - **Grafana:** [http://localhost:3400](http://localhost:3400)  
     Login with `admin` / `admin`.

7. **Batch Prediction & Monitoring**
   - After running the train flow, run the batch-predict flow using the preset parameters.
   - Run it a second time, changing the number `2` to `3` to use another dataset.
   - Results will be stored in the Postgres database.
   - Run the Evidently monitoring flow to check for data drift.
     - Results are stored in the database and as HTML in `monitoring/metrics`.

> Everything should work out of the box if you follow these steps. If you encounter issues, check Docker logs and ensure all environment variables are set correctly.
>  
�