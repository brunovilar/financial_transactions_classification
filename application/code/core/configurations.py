from typing import List

from pydantic import BaseModel, BaseSettings


class DatasetsConfiguration(BaseModel):
    external_dataset: str
    base_path: str


class MLFlowConfiguration(BaseModel):
    uri: str
    experiment_name: str
    base_model_name: str
    modified_model_name: str


class ModelTrainingConfiguration(BaseModel):
    folds: int
    min_validation_size: int
    model_seed: int
    data_seed: int


class PlotsConfiguration(BaseModel):
    large_fig_size: List[int]
    small_fig_size: List[int]
    font_size: int


class Environment(BaseSettings):
    datasets: DatasetsConfiguration
    mlflow: MLFlowConfiguration
    model_training: ModelTrainingConfiguration
    plots: PlotsConfiguration


configs = Environment()
