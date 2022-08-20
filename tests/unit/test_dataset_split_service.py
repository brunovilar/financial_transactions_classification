import pandas as pd
from pandas.testing import assert_frame_equal

from application.code.core import dataset_split_service as subject


def test_compute_cumulative_records_by_date():

    input_df = pd.DataFrame(
        [
            {"coluna_a": 1, "data": "1.1.2022", "coluna_b": "record 1"},
            {"coluna_a": 2, "data": "1.1.2022", "coluna_b": "record 2"},
            {"coluna_a": 3, "data": "2.1.2022", "coluna_b": "record 3"},
            {"coluna_a": 4, "data": "3.1.2022", "coluna_b": "record 4"},
            {"coluna_a": 5, "data": "3.1.2022", "coluna_b": "record 5"},
            {"coluna_a": 6, "data": "4.1.2022", "coluna_b": "record 6"},
            {"coluna_a": 7, "data": "5.1.2022", "coluna_b": "record 7"},
            {"coluna_a": 8, "data": "6.1.2022", "coluna_b": "record 8"},
            {"coluna_a": 9, "data": "6.1.2022", "coluna_b": "record 9"},
            {"coluna_a": 10, "data": "6.1.2022", "coluna_b": "record 10"},
        ]
    )
    input_df = input_df.sample(frac=1.0)  # Shuffle data to assert final order

    actual_df = subject.compute_cumulative_records_by_date(input_df)

    expected_df = pd.DataFrame(
        [
            {
                "period": "2022-01-01",
                "transactions": 2,
                "total_transactions": 2,
                "percentage": 20.0,
            },
            {
                "period": "2022-01-02",
                "transactions": 1,
                "total_transactions": 3,
                "percentage": 30.0,
            },
            {
                "period": "2022-01-03",
                "transactions": 2,
                "total_transactions": 5,
                "percentage": 50.0,
            },
            {
                "period": "2022-01-04",
                "transactions": 1,
                "total_transactions": 6,
                "percentage": 60.0,
            },
            {
                "period": "2022-01-05",
                "transactions": 1,
                "total_transactions": 7,
                "percentage": 70.0,
            },
            {
                "period": "2022-01-06",
                "transactions": 3,
                "total_transactions": 10,
                "percentage": 100.0,
            },
        ]
    )
    assert_frame_equal(actual_df, expected_df)
