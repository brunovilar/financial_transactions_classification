from pydantic import BaseModel, BaseSettings


class DatasetsConfiguration(BaseModel):
    external_dataset: str
    base_path: str


class MLFlowConfiguration(BaseModel):
    uri: str
    experiment_name: str


class ModelTrainingConfiguration(BaseModel):
    folds: int
    min_validation_size: int
    random_seed: int


class Environment(BaseSettings):
    datasets: DatasetsConfiguration
    mlflow: MLFlowConfiguration
    model_training: ModelTrainingConfiguration


configs = Environment()
