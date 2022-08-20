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
