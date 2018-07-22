# -*- coding: utf-8 -*-

"""
Procedures for training and loading the renewal prediction models.

General strategy: tuning by grid search for individual models, then build
a weighted ensemble from them all.

This script can be called directly with

    python predictor.py [model_id]

which will train the model specified.  This train procedure is automatically
called when main.py is run.
"""

import os
import sys
import pickle
import warnings
import json
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
from data import get_data
from typing import Dict, Tuple, Any, Union


os.chdir(os.path.dirname(__file__))


# Settings
n_cv_folds = 3
verbosity = 10
run_nested_cv = False
model_save_dir = './save/models/'


def get_params(
    model_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Get the pre-defined parameters for a given model.

    This is where all fixed parameters are defined.  Search parameters to be
    used in grid searching can also be stated here.

    Args:
        model_id: The model identifier e.g. 'xgb', 'mlp', ...
        X_train:  The training dataset features.
        y_train:  The training dataset targets.

    Returns:
        3-tuple of (model class, fixed params, search params)

    """

    # Some data stats
    total_count = len(y_train)
    pos_count = np.count_nonzero(y_train)
    # pos_prob = pos_count / total_count
    neg_count = total_count - pos_count
    # neg_prob = 1 - pos_prob
    neg_pos_ratio = neg_count / pos_count

    #
    # Ensemble
    #

    if model_id == 'ensemble':
        clf = VotingClassifier
        fixed_params = {
            'estimators': [
                m('xgb'),
                m('ada'),
                m('knn'),
                m('qda'),
            ],
            'voting': 'soft',
            'n_jobs': -1,
            'weights': [100, 5, 50, 10]
        }
        search_params = {}

    #
    # XGBoost
    #

    elif model_id == 'xgb':  # tuned
        clf = XGBClassifier
        fixed_params = {
            'silent': True,  # set False for more verbosity
            'eval_metric': 'auc',
            'n_jobs': -1,
            # tuned:
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'scale_pos_weight': neg_pos_ratio,
            'max_depth': 4,
            'min_child_weight': 9,
            'gamma': 0.4,
            'subsample': 0.85,
            'colsample_bytree': 0.55,
            'reg_alpha': 0.09,
            'reg_lambda': 0.95,
        }
        search_params = {}

    #
    # AdaBoost
    #

    elif model_id == 'ada':
        clf = AdaBoostClassifier
        fixed_params = {
            # tuned:
            'base_estimator':
                DecisionTreeClassifier(max_depth=3, min_samples_leaf=1,
                                       min_samples_split=2,
                                       class_weight='balanced'),
            'algorithm': 'SAMME.R',
            'n_estimators': 40,
            'learning_rate': 0.1,
        }
        search_params = {}

    #
    # KNN
    #

    elif model_id == 'knn':
        clf = KNeighborsClassifier
        fixed_params = {
            'weights': 'uniform',
            'p': 2,
            'n_jobs': -1
        }
        search_params = {
            'n_neighbors': [1000],
        }

    #
    # MLP/neural net
    #

    elif model_id == 'mlp':  # tuned
        clf = MLPClassifier
        h_dim = int(round((X_train.shape[1] / 1)))
        fixed_params = {
            'solver': 'adam',
            # tuned:
            'hidden_layer_sizes': (h_dim, h_dim)
        }
        search_params = {}

    #
    # SVM
    #

    elif model_id == 'svm':
        clf = SVC
        fixed_params = {
            'probability': True,
            'kernel': 'rbf',
            'class_weight': 'balanced',
            'C': 1.0
        }
        search_params = {}

    #
    # Naive Bayes
    #

    elif model_id == 'gnb':  # tuned
        clf = GaussianNB
        fixed_params = {}
        search_params = {}

    #
    # Discriminant analysis
    #

    elif model_id == 'qda':  # tuned
        clf = QuadraticDiscriminantAnalysis
        fixed_params = {
            'reg_param': 0.0
        }
        search_params = {}

    else:
        raise ValueError('Model ID [%s] is invalid' % model_id)

    return clf, fixed_params, search_params


def train_model(
    model_id: str = 'ensemble',
    include_info: bool = False
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """Train a model.

    Args:
        model_id:     The identifier of the model to train e.g. mlp, xgb, ...
        include_info: Whether to return information about the model and training
                      session.

    Returns:
        Either the trained model, or (trained model, information) if
        include_info is True.

    """

    print('Training renewal predictor model - %s' % model_id)

    # Load the data
    X_train, y_train, _, _, _ = get_data()

    # Get classifier and params
    clf, fixed_params, search_params = get_params(model_id, X_train, y_train)

    # Current version of sklearn uses a depreciated API
    # Won't be in main until Aug 2018 so just suppress for now
    # https://github.com/scikit-learn/scikit-learn/issues/10449
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    # Metric storage
    metrics = {}

    if len(search_params.keys()) > 0:

        print('\tGrid searching over parameters:')
        print('\t\t%s' % search_params)

        # Run grid search
        model = GridSearchCV(
            clf(**fixed_params),
            cv=n_cv_folds,
            param_grid=search_params,
            scoring='roc_auc',
            verbose=verbosity,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Report results
        metrics['inner_cv_auc'] = model.best_score_
        print('\tBest parameters found:')
        print('\t\t%s' % model.best_params_)
        print('\tROC-AUC (Inner CV): %s' % metrics['inner_cv_auc'])
        params = {**fixed_params, **model.best_params_}

        if run_nested_cv:
            # Do an 'outer' cross validation to give a more realistic eval
            print('\tRunning outer CV for evaluation')
            outer_cv = KFold(n_splits=n_cv_folds, shuffle=True)
            metrics['outer_cv_auc'] = cross_val_score(model, n_jobs=-1,
                                                      X=X_train,
                                                      y=y_train,
                                                      cv=outer_cv).mean()
            print('\tROC-AUC (Outer CV): %s' % metrics['outer_cv_auc'])

    else:
        # No param search, just use those given
        params = fixed_params

    # Rebuild the model with the chosen params
    model = clf(**params)
    print('(Re)training final model on full train set')

    # Note fit() does nothing for ensemble
    model.fit(X_train, y_train)
    metrics['train_auc'] = roc_auc_score(y_train,
                                         model.predict_proba(X_train)[:, 1])
    print('\tROC-AUC (Train): %s' % metrics['train_auc'])

    # Return appropriately
    if include_info:
        return model, {'params': params, 'metrics': metrics}
    else:
        return model


def get_model(
    model_id: str = 'ensemble',
    include_info: bool = False
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """Load and return the renewal predictor model.

    From disk, if found, or trained from scratch otherwise.

    Args:
        model_id:     The identifier of the model to load e.g. mlp, xgb, ...
        include_info: Whether to return information about the model and training
                      session.

    Returns:
        Either the trained model, or (trained model, information) if
        include_info is True.

    """
    this_model_save_dir = os.path.join(model_save_dir, '%s/' % model_id)
    model_file_path = os.path.join(this_model_save_dir, 'model.pickle')
    model_info_file_path = os.path.join(this_model_save_dir, 'info.json')
    if os.path.isfile(model_file_path):
        print('Loading \'%s\' renewal predictor model from disk [%s]' %
              (model_id, this_model_save_dir))
        print('Delete or rename this directory to allow retraining')
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        with open(model_info_file_path, 'r') as f:
            info = json.load(f)
    else:
        model, info = train_model(model_id, include_info=True)
        # Save to disk
        print('Saving model to %s' % this_model_save_dir)
        if os.path.isfile(model_file_path):
            os.unlink(model_file_path)
        if os.path.isfile(model_info_file_path):
            os.unlink(model_info_file_path)
        ensure_dir(this_model_save_dir)
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)
        try:
            info_json = json.dumps(info)
        except TypeError:
            info_json = '{}'
        with open(model_info_file_path, 'w') as f:
            f.write(info_json)

    # Return appropriately
    if include_info:
        return model, info
    else:
        return model


def m(model_id: str) -> Tuple[str, Any]:
    """Utility function to get a model tuple as required for sklearn's ensemble
    class.

    Args:
        model_id: The model identifier for which to create a tuple.

    Returns:
        The tuple (model ID, model object).

    """
    return model_id, get_model(model_id)


def ensure_dir(path: str) -> None:
    """Utility function for ensuring that a directory exists.

    Args:
        path: The directory to ensure exists.

    Returns:
        None

    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    get_model(sys.argv[1])
