import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.client import MlflowClient
from mlflow.pyfunc import PythonModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.fluent import ActiveRun as MLFlowRun


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


def register_model(model: PythonModel, run: MLFlowRun, model_name: str) -> int:

    mlflow.start_run(run.info.run_id)
    mlflow.sklearn.log_model(model, model_name)

    client = MlflowClient()

    if not client.get_registered_model(model_name):
        client.create_registered_model(model_name)

    runs_uri = f"runs:/{run.info.run_id}/model_name"
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

    model_version = client.create_model_version(model_name, model_src, run.info.run_id)

    return model_version.version


def publish_model(model_name: str, model_version: int, stage: str):

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=stage
    )
