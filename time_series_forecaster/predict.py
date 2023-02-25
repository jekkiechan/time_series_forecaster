import argparse
import pandas as pd
from time_series_forecaster.forecaster import Transformer, LightGBMForecaster


def main(train_file: str, test_file: str, output_file: str):

    hyperparams = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 128,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 2023,
    }

    DATE_COL = "first_day_of_month"
    TARGET_COL = "microbusiness_density"
    ID_COL = "cfips"
    FEATURE_COLS = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
        "lag_6",
        "lag_7",
        "lag_8",
        "lag_9",
        "lag_10",
        "lag_11",
        "lag_12",
        "moving_window_4_mean",
        "moving_window_4_max",
        "moving_window_4_min",
        "moving_window_4_std",

    ]
    OUTPUT_COLS = [
        "row_id",
        "microbusiness_density",
    ]

    # Load train and test data
    train_data = pd.read_csv(train_file, parse_dates=[DATE_COL])
    test_data = pd.read_csv(test_file, parse_dates=[DATE_COL])

    # Fit transformer and model
    transformer = Transformer(
        id_col=ID_COL,
        target_col=TARGET_COL,
        nb_last_values=12,
        lags=12,
        window=4,
    )

    train_data = transformer.fit(train_data)

    model = LightGBMForecaster(
        id_col=ID_COL,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        hyperparams=hyperparams,
        transformer=transformer,
    )

    model.fit(train_data[FEATURE_COLS], train_data[TARGET_COL])

    # Generate predictions
    predictions = model.predict(test_data, FEATURE_COLS)

    # Write predictions to output file
    predictions = predictions.merge(test_data[["row_id", DATE_COL, ID_COL]], on=[DATE_COL, ID_COL])
    predictions[OUTPUT_COLS].to_csv(output_file, index=False)


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path to train data file')
    parser.add_argument('--test', type=str, required=True, help='Path to test data file')
    parser.add_argument('--output', type=str, required=True, help='Path to output predictions file')
    args = parser.parse_args()

    # Run main function with input arguments
    main(train_file=args.train, test_file=args.test, output_file=args.output)
