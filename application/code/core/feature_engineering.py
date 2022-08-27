from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

LABELS_MAP = {
    "servi\x82o": "serviço",
    "farmacias": "farmácias",
    "m.o.t.o.": "compra online",
    "loja de depart": "loja de departamento",
    "vestuario": "vestuário",
    "moveis e decor": "móveis e decoração",
    "hosp e clinica": "hospitais e clínicas",
    "mat construcao": "materiais construção",
    "posto de gas": "posto de gás",
    "cia aereas": "companhia aérea",
    "trans financ": "transação financeira",
    "hoteis": "hotéis",
    "auto pe\x82as": "auto-peças",
    "agencia de tur": "agência turismo",
    "alug de carros": "aluguel carro",
}


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
            cidade_diferente=lambda f:
                f["cidade"] != f["cidade_estabelecimento"]
        )
        .assign(
            estado_diferente=lambda f:
                f["estado"] != f["estado_estabelecimento"]
        )
        .assign(pais_diferente=lambda f:
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
        .assign(dia_semana=lambda f: f["period_date"].dt.dayofweek)
        .assign(dia_mes=lambda f: f["period_date"].dt.day)
        .assign(mes=lambda f: f["period_date"].dt.month)
        .assign(dia_util=lambda f: ~f["dia_semana"].isin([5, 6]))
        .drop(columns=['period_date'])
    )
    # fmt: on


def compute_relative_value(df: pd.DataFrame) -> pd.DataFrame:

    total_relative_value = df["valor"] / df["limite_total"]
    available_relative_value = df["valor"] / df["limite_disp"]

    # fmt: off
    return (
        df
        .assign(valor_relativo_total=total_relative_value)
        .assign(valor_relativo_disponivel=available_relative_value)
    )
    # fmt: on


def transform_gender(df: pd.DataFrame) -> pd.DataFrame:

    return df.assign(sexo=lambda f: f["sexo"] == "f")


def standardize_label(label: str) -> str:

    clean_label = label.strip().lower()

    return LABELS_MAP.get(clean_label, clean_label)


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:

    fixed_labels = df.grupo_estabelecimento.apply(standardize_label)

    return df.assign(grupo_estabelecimento=fixed_labels)


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
