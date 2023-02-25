import pandas as pd
import numpy as np
import pytest

from time_series_forecaster.forecaster import Transformer


@pytest.fixture
def test_df_fit():
    df = pd.DataFrame(
        {
            "date": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
            ],
            "category": ["dog", "dog", "dog", "dog", "cat", "cat", "cat", "cat"],
            "search_volume": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    return df


@pytest.fixture
def test_df_transform():
    df = pd.DataFrame(
        {
            "date": [
                "2022-01-05",
                "2022-01-06",
                "2022-01-05",
                "2022-01-06",
            ],
            "category": ["dog", "dog", "cat", "cat"],
            "search_volume": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    return df


def test_fit(test_df_fit):
    transformer = Transformer(
        id_col="category",
        nb_last_values=2,
        lags=2,
        target_col="search_volume",
    )

    transformer.fit(test_df_fit)

    expected_output = pd.DataFrame(
        {
            "date": ["2022-01-03", "2022-01-04", "2022-01-03", "2022-01-04"],
            "category": ["dog", "dog", "cat", "cat"],
            "search_volume": [3, 4, 7, 8],
            "lag_1": [2, 3, 6, 7],
            "lag_2": [1, 2, 5, 6],
        }
    )

    pd.testing.assert_frame_equal(
        transformer.last_values.reset_index(drop=True),
        expected_output,
        check_index_type=False,
        check_dtype=False,
    )


def test_transform(test_df_fit, test_df_transform):
    transformer = Transformer(id_col="category", nb_last_values=2, lags=2, target_col="search_volume",)
    transformer.fit(test_df_fit)

    output = transformer.transform(test_df_transform)
    expected_output = pd.DataFrame(
        {
            "date": [
                "2022-01-05",
                "2022-01-06",
                "2022-01-05",
                "2022-01-06",
            ],
            "category": ["dog", "dog", "cat", "cat"],
            "search_volume": [np.nan, np.nan, np.nan, np.nan],
            "lag_1": [4, np.nan, 8, np.nan],
            "lag_2": [3, 4, 7, 8],
        }
    )

    pd.testing.assert_frame_equal(
        output.reset_index(drop=True),
        expected_output,
        check_index_type=False,
        check_dtype=False,
    )
