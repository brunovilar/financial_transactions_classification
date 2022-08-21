from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics


def compute_multiclass_classification_metrics(
    y_train: np.ndarray, y_preds: np.ndarray, average=None
) -> Dict:

    return {
        "precision": metrics.precision_score(
            y_train, y_preds, average=average, zero_division=0
        ),
        "recall": metrics.recall_score(
            y_train, y_preds, average=average, zero_division=0
        ),
        "f1": metrics.f1_score(y_train, y_preds, average=average, zero_division=0),
    }


def generate_feature_importance_report(
    model: LGBMClassifier, columns: List[str]
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {"feature": columns, "absolute_importance": model.feature_importances_}
        )
        .sort_values(by="absolute_importance", ascending=False)
        .assign(
            relative_importance=lambda f: (
                f["absolute_importance"] / f["absolute_importance"].sum() * 100
            ).apply(lambda i: f"{i:.2f}%")
        )
    )
