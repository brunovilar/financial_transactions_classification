from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:

    fixed_columns = {column: column.strip() for column in df.columns}

    return df.rename(columns=fixed_columns)


def format_value_column(raw_value: str) -> float:

    # fmt: off
    str_value = (
        str(raw_value)
        .strip()
        .replace(".", "")
        .replace(",", ".")
        .replace("-", "")
    )
    # fmt: on

    if len(str_value) > 0:
        return float(str_value)

    return np.nan


def format_string_column(raw_value: str) -> str:

    value = raw_value or ""

    return value.lower().strip()


def format_string_columns(df: pd.DataFrame, columns=List[str]) -> pd.DataFrame:

    for column in columns:
        df[column] = df[column].apply(format_string_column)

    return df


def change_column_types(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df
        .assign(valor=lambda f: f["valor"].apply(format_value_column))
        .astype({"id": str})
    )
    # fmt: on


def fill_establishment_state_from_city(df: pd.DataFrame) -> pd.DataFrame:

    city_state_map = (
        df[["cidade", "estado"]]
        .drop_duplicates()
        .set_index("cidade")
        .to_dict()["estado"]
    )

    # fmt: off
    return df.assign(
        estado_estabelecimento=lambda f:
            f["cidade_estabelecimento"].map(city_state_map)
    )
    # fmt: on


def compute_travelling_features(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df.assign(
            is_a_different_city=lambda f:
                f["cidade"] != f["cidade_estabelecimento"]
        )
        .assign(
            is_a_different_state=lambda f:
                f["estado"] != f["estado_estabelecimento"]
        )
        .assign(is_a_different_country=lambda f:
                f["pais_estabelecimento"] != "br")
    )
    # fmt: on


def compute_date_features(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df.assign(
            period_date=lambda f:
                f["period"].apply(lambda p: datetime.fromisoformat(p))
        )
        .assign(weekday=lambda f: f["period_date"].dt.dayofweek)
        .assign(monthday=lambda f: f["period_date"].dt.day)
        .assign(month=lambda f: f["period_date"].dt.month)
    )
    # fmt: on


def compute_relative_value(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df
        .assign(total_relative_value=lambda f:
                f["valor"] / f["limite_total"])
        .assign(available_relative_value=lambda f:
                f["valor"] / f["limite_disp"])
    )
    # fmt: off


def transform_gender(df: pd.DataFrame) -> pd.DataFrame:

    return df.assign(genero=lambda f: f["sexo"] == "f")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # fmt: off
    return (
        df
        .pipe(fill_establishment_state_from_city)
        .pipe(compute_travelling_features)
        .pipe(compute_date_features)
        .pipe(transform_gender)
        .pipe(compute_relative_value)
    )
    # fmt: on
