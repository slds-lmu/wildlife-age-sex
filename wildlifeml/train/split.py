import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(
    data: pd.DataFrame,
    stratify_by: str | None = None,
    train_prop: float = 0.8,
    num_splits: int = 1,
):
    """
    Split the data into train and test sets.

    Args:
    -----
    data: pd.DataFrame
        Dataframe to split.
    stratify_by: Optional[list[str]]
        The columns to stratify by if possible.
    proportions: list[float]
        The proportions of each sample in the split, nust sum to 1.
    num_splits: int
        The number of different splits to make. Split more than once for cross-validation.

    Yields:
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        A tuple of dataframes, train and test.
    """

    # Validate stratify_by
    if stratify_by:
        if stratify_by not in data.columns:
            raise ValueError("stratify_by must be a column in the data.")
        logging.warning(
            f"Dropping {len(data[data[stratify_by].isna()])} rows with missing values in `{stratify_by}`."
        )
        data = data.dropna(subset=[stratify_by])

    # Stratify by specified column or do not stratify
    groups_to_stratify_by = data[stratify_by] if stratify_by else np.ones(len(data))

    sss = StratifiedShuffleSplit(n_splits=num_splits, train_size=train_prop)
    try:
        logging.info(f"Trying to split data stratified by `{stratify_by}`.")
        splits = sss.split(data, groups_to_stratify_by)
    except ValueError as e:
        logging.warning(
            f"StratifiedShuffleSplit failed with error: {e} Returning unstratified split."
        )
        splits = sss.split(data, np.ones(len(data)))

    for train_data_idx, test_data_idx in splits:
        yield data.iloc[train_data_idx], data.iloc[test_data_idx]
