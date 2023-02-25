import numpy as np
from sklearn.model_selection import BaseCrossValidator


class CustomTimeSeriesCV(BaseCrossValidator):
    """Custom cross-validation for time series data with multiple groups.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds.

    date_col : str, default='date'
        Name of the date column in the data.

    id_col : str, default='id'
        Name of the ID column in the data.

    target_col : str, default='target'
        Name of the target column in the data.
    """

    def __init__(self, n_splits=5, date_col='date', id_col='id', target_col='target'):
        self.n_splits = n_splits
        self.date_col = date_col
        self.id_col = id_col
        self.target_col = target_col

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and validation sets.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.

        y : pandas.Series, default=None
            Ignored. Present here for API consistency by convention.

        groups : pandas.Series, default=None
            The grouping of the data. Each unique value in the grouping column
            is treated as a separate group.

        Yields
        ------
        train_index : numpy.ndarray
            The indices of the training data.

        test_index : numpy.ndarray
            The indices of the validation data.
        """
        # Get unique values of the grouping column
        groups = X[self.id_col].unique()

        # Sort the data by the date column within each group
        X = X.sort_values([self.id_col, self.date_col])

        # Calculate the size of each fold
        fold_size = len(groups) // self.n_splits

        # Iterate over the folds
        for i in range(self.n_splits):
            # Calculate the start and end indices of the fold
            start = i * fold_size
            end = (i + 1) * fold_size

            # Get the groups for the validation data in this fold
            val_groups = groups[start:end]

            # Split the data into training and validation sets based on the grouping
            train_idx = X[X[self.id_col].isin(groups) & ~X[self.id_col].isin(val_groups)].index.values
            val_idx = X[X[self.id_col].isin(val_groups)].index.values

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def _cv_score(self, preds, train_data):
        """Custom evaluation metric for cross-validation."""
        y_true = train_data.get_label()
        
        score = np.sqrt(np.mean(np.square(np.log1p(preds) - np.log1p(y_true)))))
        return ('rmsle', score, False)


