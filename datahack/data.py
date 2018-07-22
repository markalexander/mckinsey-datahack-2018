# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler


def get_data() -> Tuple[np.ndarray, np.ndarray,
                        pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and pre-process the data.

    Returns:
        X_train, y_train, df_train, X_test, df_test

    """

    # Load the data
    df_train = pd.read_csv('./data/train.csv', header=0)
    df_test = pd.read_csv('./data/test.csv', header=0)

    # Select features
    use_cols = [
        'perc_premium_paid_by_cash_credit',
        'age_in_days',
        'Income',
        'Count_3-6_months_late',
        'Count_6-12_months_late',
        'Count_more_than_12_months_late',
        'application_underwriting_score',
        'no_of_premiums_paid',
        'sourcing_channel',
        'residence_area_type'
    ]

    # Combine train/test for processing, we'll re-split them later
    X = df_train[use_cols].append(df_test[use_cols])

    # Categorical features to integer labels
    labeler = LabelEncoder()
    for col_name in ['sourcing_channel', 'residence_area_type']:
        X[col_name] = labeler.fit_transform(X[col_name])

    # Impute missing values
    imputer = Imputer(strategy='median')
    X = imputer.fit_transform(X.values)

    # Re-split into train/test and convert to ndarray
    X_train = X[0:df_train.shape[0]]
    X_test = X[df_train.shape[0]::]
    y_train = df_train['renewal'].values

    # Only useful for some models, but doesn't hurt the others regardless
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, y_train, df_train, X_test, df_test
