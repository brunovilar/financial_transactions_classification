from pydantic import BaseModel, BaseSettings


class MLFlowConfiguration(BaseModel):
    uri: str
    experiment_name: str


class Configuration(BaseModel):
    external_dataset: str
    base_path: str
    mlflow: MLFlowConfiguration


class Environment(BaseSettings):
    configs: Configuration


environment = Environment()

configs = environment.configs
