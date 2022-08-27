import copy
from functools import partial
from typing import Dict, List, Optional, Tuple

from category_encoders import CountEncoder
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from application.code.core.feature_engineering import (
    change_column_types,
    clean_column_names,
    engineer_features,
    format_string_columns,
    standardize_labels,
    transform_gender,
)


def combine_feature_columns(*args, **kwargs) -> List[str]:

    combined_columns = []

    for arg in args:
        combined_columns += arg

    for value in kwargs.values():
        combined_columns += value

    return sorted(list(set(combined_columns)))


def vectorize_folds(
    folds: List[Tuple[DataFrame, DataFrame]],
    columns_selection: List[str],
    categorical_columns: List[str],
    high_cardinality_categorical_columns: List[str],
    binary_columns: List[str],
    target_column: str,
) -> List[
    Tuple[Tuple[ndarray, List[int]], Tuple[ndarray, List[int]], List[str]]
]:  # noqa

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
        # For production, it should raise an exception
        labels = sorted(
            list(
                set(
                    clean_training_df[target_column].tolist()
                    + clean_assessment_df[target_column].tolist()  # noqa
                )
            )
        )

        features_selection = [c for c in columns_selection if c != target_column]

        label_encoder, categorical_encoder = generate_encoders(
            clean_training_df[features_selection],
            labels,
            high_cardinality_categorical_columns,
        )

        generate_fn = partial(
            generate_features_and_labels,
            columns_selection=features_selection,
            target_column=target_column,
            binary_columns=binary_columns,
            label_encoder=label_encoder,
            categorical_encoder=categorical_encoder,
        )

        training_X, training_y = generate_fn(clean_training_df)
        assessment_X, assessment_y = generate_fn(df=clean_assessment_df)

        vectorized_folds.append(
            ((training_X, training_y), (assessment_X, assessment_y), labels)
        )

    return vectorized_folds


def vectorize_dataset(
    df: DataFrame,
    label_encoder: LabelEncoder,
    categorical_encoder: CountEncoder,
    columns_selection: List[str],
    categorical_columns: List[str],
    binary_columns: List[str],
    target_column: str,
) -> Tuple[ndarray, List[int]]:

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


def clean_data(df: DataFrame, categorical_columns: List[str]) -> DataFrame:

    return (
        df.pipe(clean_column_names)
        .pipe(change_column_types)
        .pipe(format_string_columns, columns=categorical_columns)
        .pipe(standardize_labels)
    )


def generate_encoders(
    features_df: DataFrame,
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


def generate_features(
    df: DataFrame,
    columns_selection: List[str],
    binary_columns: List[str],
    categorical_encoder: CountEncoder,
) -> ndarray:

    features_df = df[columns_selection].copy()

    numeric_features_df = (
        features_df.pipe(transform_gender)
        .pipe(categorical_encoder.transform)
        .astype({c: "int" for c in binary_columns})
    )

    return numeric_features_df.to_numpy()


def generate_features_and_labels(
    df: DataFrame,
    columns_selection: List[str],
    target_column: str,
    binary_columns: List[str],
    label_encoder: LabelEncoder,
    categorical_encoder: CountEncoder,
) -> Tuple[ndarray, List[int]]:

    labels = df[target_column].tolist()
    y = label_encoder.transform(labels)

    features_selection = [c for c in columns_selection if c != target_column]

    numeric_features = generate_features(
        df, features_selection, binary_columns, categorical_encoder
    )

    return numeric_features, y


def compute_weights(y: List[int]) -> Dict[int, float]:

    n_samples = len(y)
    n_classes = len(set(y))

    compute_weight_fn = lambda r: n_samples / (n_classes * r)  # noqa

    value_counts_frame = (
        DataFrame({"label": y})
        .assign(records=1)
        .groupby(["label"])
        .sum()
        .reset_index()
        .assign(weight=lambda f: f["records"].apply(compute_weight_fn))
    )

    iterator = value_counts_frame[["label", "weight"]].itertuples(index=False)

    return {item.label: item.weight for item in iterator}


def adjust_weights(
    class_weights: Dict[int, float],
    labels: List[str],
    adjustments: Optional[Dict[int, float]],
) -> Dict[int, float]:

    adjustments: Dict[int, float] = adjustments or dict()
    adjusted_weights = copy.deepcopy(class_weights)

    for key, value in adjustments.items():

        encoded_label = labels.index(key)
        adjusted_weights[encoded_label] = adjusted_weights[encoded_label] * value

    return adjusted_weights
