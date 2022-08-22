from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking.fluent import ActiveRun as MLFlowRun

from application.code.adapters.mlflow_adapter import log_dataframe_artifact
from application.code.core.model_evaluation import (
    compute_multiclass_classification_metrics,
    generate_feature_importance_report,
)
from application.code.core.model_training import compute_weights


def train_model(
    algorithm_class: Any,
    model_params: Dict,
    X_training: np.ndarray,
    y_training: List[int],
    features_names: List[str],
    experiment_run_name: str,
    extra_artifacts: List[Tuple[pd.DataFrame, str, str]],
) -> Tuple[MLFlowRun, Any]:

    with mlflow.start_run(run_name=experiment_run_name) as mlflow_run:
        model = algorithm_class(**model_params)
        model.fit(X_training, y_training)

        # Training metrics
        preds = model.predict(X_training)
        training_metrics = compute_multiclass_classification_metrics(y_training, preds)

        # Log Parameters
        mlflow.log_params(model_params)
        class_weights = compute_weights(y_training)

        model_params.update(
            {"class_weight": class_weights, "num_class": len(set(y_training))}
        )

        # Log Main Artifacts
        class_weights_df = pd.DataFrame(
            [{"class": key, "weight": value} for key, value in class_weights.items()]
        )
        log_dataframe_artifact(class_weights_df, "main model", "class weights")

        training_metrics_df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in training_metrics.items()]
        )
        log_dataframe_artifact(training_metrics_df, "main model", "training metrics")

        features_importance_df = generate_feature_importance_report(
            model, features_names
        )
        log_dataframe_artifact(
            features_importance_df, "main model", "features importance"
        )

        # Log Extra Artifacts
        for artifact_df, artifact_folder, artifact_name in extra_artifacts:
            log_dataframe_artifact(artifact_df, artifact_folder, artifact_name)

    return mlflow_run, model
