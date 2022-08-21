from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.preprocessing import LabelEncoder

from application.code.core.feature_engineering import (
    change_column_types,
    clean_column_names,
    engineer_features,
    format_string_columns,
    transform_gender,
)


def vectorize_folds(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    columns_selection: List[str],
    categorical_columns: List[str],
    high_cardinality_categorical_columns: List[str],
    binary_columns: List[str],
    target_column: str,
) -> List[Tuple[Tuple[np.ndarray, List[int]], Tuple[np.ndarray, List[int]]]]:

    vectorized_folds = []

    for training_df, assessment_df in folds:

        # fmt: off
        clean_training_df = (
            training_df
            .pipe(clean_data, categorical_columns + [target_column])
            .pipe(engineer_features)
        )

        clean_assessment_df = (
            assessment_df
            .pipe(clean_data, categorical_columns + [target_column])
            .pipe(engineer_features)
        )
        # fmt: on

        # Labels are combined to avoid uknown label issues during experiments
        # For production, we should raise an exception
        labels = list(
            set(
                clean_training_df[target_column].tolist()
                + clean_assessment_df[target_column].tolist()  # noqa
            )
        )

        label_encoder, categorical_encoder = generate_encoders(
            clean_training_df[columns_selection],
            labels,
            high_cardinality_categorical_columns,
        )

        generate_fn = partial(
            generate_features_and_labels,
            columns_selection=columns_selection,
            target_column=target_column,
            binary_columns=binary_columns,
            label_encoder=label_encoder,
            categorical_encoder=categorical_encoder,
        )

        training_X, training_y = generate_fn(clean_training_df)
        assessment_X, assessment_y = generate_fn(df=clean_assessment_df)

        vectorized_folds.append(
            ((training_X, training_y), (assessment_X, assessment_y))
        )

    return vectorized_folds


def vectorize_dataset(
    df: pd.DataFrame,
    label_encoder: LabelEncoder,
    categorical_encoder: CountEncoder,
    columns_selection: List[str],
    categorical_columns: List[str],
    binary_columns: List[str],
    target_column: str,
) -> Tuple[np.ndarray, List[int]]:

    # fmt: off
    clean_df = (
        df
        .pipe(clean_data, categorical_columns + [target_column])
        .pipe(engineer_features)
    )
    # fmt: on

    X, y = generate_features_and_labels(
        clean_df,
        columns_selection=columns_selection,
        target_column=target_column,
        binary_columns=binary_columns,
        label_encoder=label_encoder,
        categorical_encoder=categorical_encoder,
    )

    return X, y


def clean_data(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:

    return (
        df.drop_duplicates()
        .pipe(clean_column_names)
        .pipe(change_column_types)
        .pipe(format_string_columns, columns=categorical_columns)
    )


def generate_encoders(
    features_df: pd.DataFrame,
    labels: List[str],
    high_cardinality_categorical_columns: List[str],
) -> Tuple[LabelEncoder, CountEncoder]:

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    categorical_encoder = CountEncoder(
        cols=high_cardinality_categorical_columns,
        handle_missing="value",
        handle_unknown="value",
    )

    categorical_encoder.fit(features_df)

    return label_encoder, categorical_encoder


def generate_features_and_labels(
    df: pd.DataFrame,
    columns_selection: List[str],
    target_column: str,
    binary_columns: List[str],
    label_encoder: LabelEncoder,
    categorical_encoder: CountEncoder,
) -> Tuple[np.ndarray, List[int]]:

    features_df = df[columns_selection].copy()
    labels = df[target_column].tolist()
    y = label_encoder.transform(labels)

    numeric_features_df = (
        features_df.pipe(transform_gender)
        .pipe(categorical_encoder.transform)
        .astype({c: "int" for c in binary_columns})
    )

    return numeric_features_df.to_numpy(), y


def compute_weights(y: List[int]) -> Dict[int, float]:

    n_samples = len(y)
    n_classes = len(set(y))

    value_counts_frame = (
        pd.DataFrame({"label": y})
        .assign(records=1)
        .groupby(["label"])
        .sum()
        .reset_index()
        .assign(
            weight=lambda f: f["records"].apply(lambda r: n_samples / (n_classes * r))
        )
    )

    return {
        item.label: item.weight
        for item in value_counts_frame[["label", "weight"]].itertuples(index=False)
    }
