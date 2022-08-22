import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.client import MlflowClient


def log_dataframe_artifact(
    df: pd.DataFrame, artifact_folder_name: str, artifact_name: str
):

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_file_path = Path(temp_dir) / f"{artifact_name}.html"
        df.to_html(artifact_file_path)

        mlflow.log_artifact(artifact_file_path, artifact_folder_name)


def get_mlflow_artifact_content(run_id: str, artifact_folder_name: str) -> dict:

    client = MlflowClient()

    with tempfile.TemporaryDirectory() as temp_dir:
        client.download_artifacts(run_id, artifact_folder_name, temp_dir)

        path = Path(temp_dir)
        artifacts = {}
        for file in path.glob("**/*"):
            if file.is_file():
                artifact_name = file.name.split(".")[0]
                artifacts[artifact_name] = file.read_text()

    return artifacts
