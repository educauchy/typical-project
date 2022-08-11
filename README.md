# Typical ML project architecture
Architecture of a typical ML project

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
