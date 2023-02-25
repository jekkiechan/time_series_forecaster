from abc import abstractmethod, ABCMeta
from typing import List, Dict

import pandas as pd
import lightgbm


DATE_COL = "date"
FEATURE_COLS = ["search_volume_lag_1", "search_volume_lag_2", "search_volume_lag_3"]
TARGET_COL = "search_volume"


class GlobalForecaster(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        date_col: str,
        id_col: str,
        target_col: str,
        hyperparams: Dict[str, any] = None,
        transformer=None,
    ):
        self.is_fitted = False
        self.name = name
        self.date_col = date_col
        self.id_col = id_col
        self.target_col = target_col
        self.hyperparams = hyperparams
        self.transformer = transformer

    @abstractmethod
    def _fit(self, X_train, y_train):
        pass

    @abstractmethod
    def _predict(self, X_test: pd.DataFrame):
        pass

    def fit(self, X_train, y_train):
        self._fit(X_train, y_train)
        self.is_fitted = True

    def predict(
        self,
        test_data: pd.DataFrame,
        feature_cols: List[str]
    ):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions can be made.")

        test_data = test_data.copy()

        dates = test_data[self.date_col].unique()
        current_date = pd.to_datetime(dates[0])

        for date in dates:
            curr_date_mask = test_data[self.date_col].eq(date)
            test_data = self.transformer.transform(test_data)

            test_data.loc[curr_date_mask, self.target_col] = self._predict(
                test_data.loc[curr_date_mask, feature_cols]
            )
            current_date += pd.DateOffset(months=1)

        return test_data[[self.date_col, self.id_col, self.target_col]]


class Transformer:
    def __init__(
        self,
        id_col: str,
        target_col,
        nb_last_values: int = 30,
        window: int = 7,
        lags: int = 5
    ):
        self.id_col = id_col
        self.nb_last_values = nb_last_values
        self.lags = lags
        self.window = window
        self.target_col = target_col

    def fit(self, data: pd.DataFrame):
        data = data.copy()
        data = data.pipe(self.add_lag_features).pipe(self.add_moving_window_features)

        self.last_values = data.groupby(self.id_col).tail(self.nb_last_values)

        return data

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        data = pd.concat([self.last_values, data])
        data = data.pipe(self.add_lag_features).pipe(self.add_moving_window_features)

        return data.iloc[self.last_values.shape[0] :]

    def add_lag_features(self, data):
        data = data.copy()
        for lag in range(1, self.lags + 1):
            data[f"lag_{lag}"] = data.groupby(self.id_col)[self.target_col].shift(lag)

        return data

    def add_moving_window_features(self, data):
        data = data.copy()

        group = data.groupby(self.id_col)[self.target_col]

        data[f"moving_window_{self.window}_mean"] = group.shift(1).rolling(self.window).mean()
        data[f"moving_window_{self.window}_max"] = group.shift(1).rolling(self.window).max()
        data[f"moving_window_{self.window}_min"] = group.shift(1).rolling(self.window).min()
        data[f"moving_window_{self.window}_std"] = group.shift(1).rolling(self.window).std()

        return data


class LightGBMForecaster(GlobalForecaster):
    def __init__(
        self,
        date_col: str,
        id_col: str,
        target_col: str,
        hyperparams: Dict[str, any] = None,
        transformer=None,
    ):
        super().__init__(
            name="LightGBM",
            date_col=date_col,
            id_col=id_col,
            target_col=target_col,
            hyperparams=hyperparams,
            transformer=transformer
        )

    def _fit(self, X_train, y_train):
        lgb_train = lightgbm.Dataset(
            X_train,
            label=y_train,
            # feature_name=X_train.columns,
        )
        self.model = lightgbm.train(
            params=self.hyperparams,
            train_set=lgb_train,
            num_boost_round=self.hyperparams.get("num_boost_round", 100),
        )

    def _predict(self, X_test):
        return self.model.predict(X_test)
