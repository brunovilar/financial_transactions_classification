import mlflow

from application.code.core.configurations import configs


def test_mlflow_connection():

    mlflow.set_tracking_uri(configs.mlflow.uri)
    _ = mlflow.list_experiments()
