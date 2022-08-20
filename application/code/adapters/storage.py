from pathlib import Path

import pandas as pd


def save_dataset(df: pd.DataFrame, base_path: str, stage: str, file_name: str):
    file_path = Path(base_path) / stage
    file_path.mkdir(exist_ok=True)

    df.to_parquet(file_path / f"{file_name}.parquet")


def read_dataset(base_path: str, stage: str, file_name: str) -> pd.DataFrame:

    pd.read_parquet(Path(base_path) / stage / f"{file_name}.parquet")
