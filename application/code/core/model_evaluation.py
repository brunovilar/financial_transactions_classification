from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm.sklearn import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from sklearn.metrics import classification_report

from application.code.core.feature_engineering import standardize_labels


def compute_multiclass_classification_metrics(
    y_train: Union[List[int], ndarray],
    y_preds: Union[List[int], ndarray],
    average_options: Optional[List[str]] = None,
) -> Dict:

    average_options = average_options or ["macro", "micro", "weighted"]

    computed_metrics = dict()

    for average in average_options:

        computed_metrics.update(
            {
                f"{average}_precision": metrics.precision_score(
                    y_train, y_preds, average=average, zero_division=0
                ),
                f"{average}_recall": metrics.recall_score(
                    y_train, y_preds, average=average, zero_division=0
                ),
                f"{average}_f1": metrics.f1_score(
                    y_train, y_preds, average=average, zero_division=0
                ),
            }
        )

    return computed_metrics


def generate_feature_importance_report(
    model: LGBMClassifier, columns: List[str]
) -> DataFrame:

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


def generate_confusion_matrix_report(
    y: List[int], pred: List[int], labels: List[str]
) -> DataFrame:

    encoded_labels = set(y) | set(pred)
    known_labels = {
        ix: label for ix, label in enumerate(labels) if ix in encoded_labels
    }

    cm = metrics.confusion_matrix(y, pred)
    return pd.DataFrame(cm, columns=known_labels.values()).rename(index=known_labels)


def generate_classification_report(
    y: Union[List[int], ndarray], pred: Union[List[float], ndarray], labels: List[str]
) -> DataFrame:

    columns_to_rename = {str(ix): name for ix, name in enumerate(labels)}
    columns_to_drop = ["accuracy", "macro avg", "weighted avg"]

    report = classification_report(y, pred, output_dict=True, zero_division=0)

    return (
        pd.DataFrame(report)
        .rename(columns=columns_to_rename)
        .drop(columns=columns_to_drop)
        .T
    )


def generate_labels_support(df: DataFrame) -> DataFrame:

    return (
        df.value_counts("grupo_estabelecimento")
        .to_frame()
        .reset_index()
        .rename(columns={0: "training_support"})
        .pipe(standardize_labels)
    )


def plot_folds_metrics(df: DataFrame):

    columns = [
        c
        for c in df.columns
        if (c.endswith("precision") or c.endswith("recall") or c.endswith("f1"))
    ]

    view_df = (
        df[columns]
        .melt(id_vars=[], value_vars=columns)
        .assign(average=lambda f: f["variable"].str.split("_").str[0])
        .assign(metric=lambda f: f["variable"].str.split("_").str[1])
    )

    plt.figure(figsize=(15, 5))
    ax = sns.boxplot(x="variable", y="value", hue="average", data=view_df)
    ax.set_title(
        f"Metrics Distribution by Average Type (Folds={len(df)})",
        fontdict={"fontsize": 20},
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Column Value")
    plt.show()
