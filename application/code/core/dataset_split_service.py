from datetime import datetime
from typing import List, Tuple

import pandas as pd


def compute_cumulative_records_by_date(df: pd.DataFrame) -> pd.DataFrame:

    periods = pd.to_datetime(df["data"], format="%d.%m.%Y")

    return (
        df.assign(period=periods.apply(lambda dt: str(dt.date())))
        .sort_values(by="period")[["period"]]
        .assign(transactions=1)
        .groupby("period")
        .sum()
        .reset_index()
        .assign(total_transactions=lambda f: f["transactions"].cumsum())
        .assign(
            percentage=lambda f: (
                100 * f["total_transactions"] / f["transactions"].sum()
            )
        )
    )


def generate_folds(
    df: pd.DataFrame, n_folds: int, min_validation_size: int, seed: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:

    split_period = str(datetime.today().date())
    last_dataset = df.copy()

    folds = []

    for _ in range(n_folds):

        differences_df = (
            df.loc[lambda f: f["period"] < split_period]
            .pipe(compute_cumulative_records_by_date)
            .pipe(_compute_absolute_difference)
        )

        split_period = _get_max_period_by_absolute_difference(
            differences_df, min_validation_size
        )
        fold_train_df = last_dataset.loc[lambda f: f["period"] < split_period]
        fold_valid_df = last_dataset.loc[lambda f: f["period"] >= split_period]
        last_dataset = fold_train_df

        folds.append(
            (
                fold_train_df.sample(frac=1.0, random_state=seed),
                fold_valid_df,
            )
        )

    return folds


def _compute_absolute_difference(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df
        .assign(difference=lambda f:
                f["total_transactions"].max() - f["total_transactions"])
    )
    # fmt: on


def _get_max_period_by_absolute_difference(
    df: pd.DataFrame, min_difference: int
) -> str:

    # fmt: off
    return (
        df
        .loc[lambda f: f["difference"] >= min_difference]["period"].max()
    )
    # fmt: on


def describe_datasets(
    training_df: pd.DataFrame, assessment_df: pd.DataFrame, target_column: str
):

    training_periods = set(training_df["period"].tolist())
    assessment_periods = set(assessment_df["period"].tolist())

    training_labels = set(training_df[target_column].tolist())
    assessment_labels = set(assessment_df[target_column].tolist())

    print(f' - Split Period: {assessment_df["period"].min()}')

    print(" - Training:")
    print(f"\t - Size: {len(training_df)}")
    print(f"\t - Days: {len(training_periods)}")
    print(f"\t - Labels: {len(training_labels)}")

    print(" - Assessment:")
    print(f"\t - Size: {len(assessment_periods)}")
    print(f"\t - Days: {len(assessment_periods)}")
    print(f"\t - Labels: {len(assessment_labels)}")
    print(
        " - Assessment Relative Size: "
        f"{len(assessment_df) / (len(assessment_df)+len(training_df)) * 100:.2f}%"
    )
