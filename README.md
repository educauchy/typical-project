# Typical ML project architecture

Components:
- Git (Github) for code versioning
- DVC for dataset versioning
- AWS-like storage (Minio) for artifacts storage
- MLflow for experiments tracking
- Flask / Streamlit for web app
- Github Actions and Git Hooks for CI/CD
- Docker for containers
- Dockerhub for containers storage
- Airflow for scheduling

Minio setup:
```shell
export MINIO_CONFIG_ENV_FILE=/etc/default/minio
minio server ./mlflow
```

DVC setup for Minio:
```shell
pip install dvc[s3]
dvc remote add -d minio s3://data -f
dvc remote modify minioendpointurl http://127.0.0.1:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

dvc add data/raw/winequality-red.csv
dvc push -r minio data/raw/winequality-red.csv.dvc 
```

MLflow setup for Minio:
```shell
export AWS_ACCESS_KEY_ID='minioadmin'
export AWS_SECRET_ACCESS_KEY='minioadmin'
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export MLFLOW_ARTIFACT_LOCATION=s3://mlflow
export MLFLOW_EXPERIMENT_NAME='wine_quality_red_minio_3'
export MLFLOW_MODEL_NAME='wine_quality_red_minio_3'
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root minio/ \
              --artifacts-destination s3://mlflow \
              --host 127.0.0.1 --port 5558
venv/bin/python3.7 src/models/retrain.py
```


Here are some useful MLflow CLI commands
```shell
# mlflow cli help 
mlflow --help

# print all runs of certain experiment
mlflow runs list --experiment-id 13

# run tracking server
export MLFLOW_TRACKING_URI=http://localhost:5558

# hint: we need to launch tracking server in order to access Model Registry
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models/mlruns --host 0.0.0.0 --port 5558

# run training
mlflow run ./src/models/sklearn --entry-point main

# run UI for local usage
mlflow ui --backend-store-uri='./models/mlruns' --host 0.0.0.0 --port

# run model as REST API
mlflow models serve --model-uri "runs:/8702524acc6943ed9001fa446bc45fb1/model" --port 5557

# run prediction
mlflow run ./src/models/sklearn --entry-point predict

# run prediction with runs
mlflow models predict --model-uri 'runs:/8702524acc6943ed9001fa446bc45fb1/model' --input-path './data/processed/winequality-red-scoring.csv' --output-path './data/processed/winequality-red-scoring-result.csv' --content-type 'csv'

# run prediction with models
mlflow models predict --model-uri 'models:/logistic_regression/None' --input-path './data/processed/winequality-red-scoring.csv' --output-path './data/processed/winequality-red-scoring-result.csv' --content-type 'csv'

# run predictions with CURL
curl "http://localhost:5557/invocations" \
-H 'Content-Type: application/json' \
-d '{"columns": ["id", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "data": [[1000001, 6.6, 0.52, 0.04, 2.2, 0.069, 8.0, 15.0, 0.9956, 3.4, 0.63, 9.4]]}'
```

Here are some useful sqlite3 commands
```shell
'.headers ON' > ~/.sqliterc
'.mode columns' >> ~/.sqliterc

sqlite3 mlflow.db
.databases # display databases
.tables    # display tables

sqlite3 mlflow.db 'select * runs where experiment_id = 13'
```
