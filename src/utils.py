import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from collections import OrderedDict
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_curve, roc_auc_score,
    precision_recall_curve, log_loss)
import optuna

from src.model_paths import SCHEMA_DIR

def correlations(df, do_not_consider, ax = None):
    """
    Triangular correlation plot for df.

    :param df: dataframe with data
    :param do_not_consider: columns not to consider
    :param ax : plt.Axes, optional
    :return:
    """
    types = {col: str(df[col].dtype) for col in df.columns}
    columns = [item for item in types if types[item] not in ('object', 'str') and item not in do_not_consider]
    mask = np.triu(np.ones_like(df[columns].corr()))

    # plotting a triangle correlation heatmap
    if ax is None:
        fig, ax = plt.subplots(1, 1, )

    dataplot = sns.heatmap(df[columns].corr().round(2),
                           cmap="coolwarm",
                           annot=False,
                           mask=mask,
                           ax=ax,
                           vmax=1, vmin=-1)
    return dataplot

def load_data_schema(data_schema_path):
     """
     Load a data schema from a JSON file and return it as an ordered dictionary.

     Args:
         data_schema_path (str): The file path to the JSON data schema.

     Returns:
         OrderedDict: The data schema loaded from the JSON file, preserving the order of keys.
     """
     with open(data_schema_path, 'r', encoding='utf-8') as file:
         data_dict = json.load(file, object_pairs_hook=OrderedDict)
     return data_dict


def convert_to_numeric(value):
    """
    Convert a value to a numeric type. If conversion fails, return NaN.

    Parameters:
    -----------
    value : any
        The value to be converted.

    Returns:
    --------
    float
        The numeric representation of the value, or NaN if conversion fails.
    """
    try:
        return pd.to_numeric(value)
    except (ValueError, TypeError):
        return np.nan
    # Dataframe


def convert_numerical_features(df, columns):
    """
    Converts numeric features of dataframe to float 64
    """
    df[columns] = df[columns].apply(convert_to_numeric)
    return df


def clean_up(s):
    s = s.astype(str)
    s = s.str.upper()
    s = s.str.strip()
    s = s.str.replace(r'\s+', '_', regex=True)
    s = s.replace(r'\.0$', '', regex=True)
    s = s.replace(r'^NAN$|^$', np.nan, regex=True)
    return s


def convert_categorical_features(df, columns):
    """
    Converts numeric features of dataframe to float 64
    """
    df[columns] = df[columns].astype(
        object,
        errors='ignore'
    )
    df[columns] = df[columns].fillna('')
    df[columns] = df[columns].apply(clean_up)

    return df


def get_feature_lists(data_dict):
    """
     Extract feature lists and imputation rules from a data schema dictionary.

     This function categorizes features into numerical, boolean, and categorical types
     and identifies additional imputation and capping rules based on the provided schema.

     Args:
         data_dict (dict): A dictionary where keys are feature names and values are lists
                           containing attributes of the features. The attributes include
                           information such as type (e.g., 'numerical', 'boolean'),
                           imputation strategy, and capping values.

    Returns:
         tuple: A tuple containing the following elements:
             - numerical_features (list): List of numerical feature names.
             - categorical_features (list): List of categorical feature names.
             - feature_names (list): List of all feature names.
     """
    features = {
        "numerical"  : [
        feature for feature, attributes in data_dict.items()
        if attributes['type'] == 'numeric'
    ],
     "categorical" : [
        feature for feature, attributes in data_dict.items()
        if attributes['type'] == 'categorical'
    ]}

    impute_values = list(set(attributes["impute_value"] for attributes in data_dict.values()))
    features["impute"] = {value : [
            feature for feature, attributes in data_dict.items()
            if attributes['impute_value'] == value
               ]
            for value in impute_values}


    for value in impute_values:
        features["impute"][value] = [
            feature for feature, attributes in data_dict.items()
            if attributes['impute_value'] == value
        ]
    features["max"] = {feature: json.loads(attributes['range'])[1]
                       for feature, attributes in data_dict.items()
                       if attributes['type'] == 'numeric' and json.loads(attributes['range'])[1] is not None

                       }
    features["min"] = {feature: json.loads(attributes['range'])[0]
                       for feature, attributes in data_dict.items()
                       if attributes['type'] == 'numeric' and json.loads(attributes['range'])[0] is not None

                       }
    feature_names = list(data_dict.keys())

    return (
        features,
        feature_names
    )

def recast_numeric(df, columns):
    df = df.copy()
    df[columns] = df[columns].astype(float)
    return df

def is_jsonable(x):
    """
    Check if an object can be serialized to JSON.
     Parameters:
    ----------
    x : Any
        The object to be checked for JSON serializability.

    Returns:
    -------
    bool
        `True` if the object can be serialized to JSON, `False` otherwise.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

_INT_PARAMS = {"num_parallel_tree", "max_depth"}

# XGBoost is tuned in two stages: tree structure first, then boosting / regularization.
_XGB_STAGE_PARAMS = {
    1: ["booster", "num_parallel_tree", "max_depth", "min_child_weight", "gamma",
        "max_delta_step", "subsample", "colsample_bytree", "colsample_bylevel"],
    2: ["learning_rate", "reg_alpha", "reg_lambda", "scale_pos_weight"],
}
_XGB_DEFAULTS = {
    "booster": "gbtree", "num_parallel_tree": 1, "max_depth": 6, "min_child_weight": 1.0,
    "gamma": 0.0, "max_delta_step": 0.0, "subsample": 1.0, "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0, "learning_rate": 0.1, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
}


def load_param_grid(model_key, path=None):
    """Search-space definition for `model_key` ('xgb', 'cat' or 'lgb') from schema/param_grid.json."""
    path = path or SCHEMA_DIR / "param_grid.json"
    with open(path, "r", encoding="utf-8") as file:
        grids = json.load(file)
    if model_key not in grids:
        raise KeyError(f"No '{model_key}' section in {path}; available: {list(grids)}")
    return grids[model_key]


def _suggest_from_grid(trial, name, spec):
    """trial.suggest_* for one entry of the param grid ({low, high, step, log} or a list of choices)."""
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if name in _INT_PARAMS:
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step") or 1))
    return trial.suggest_float(name, spec["low"], spec["high"],
                               step=spec.get("step"), log=bool(spec.get("log")))


def _cv_score_with_overfit_guard(fit_predict_fn, X, y, metric, tol, n_splits, random_state=None, trial=None):
    """
    Shared CV loop for the Optuna objective functions below.

    Runs StratifiedKFold, calls `fit_predict_fn(X_train, y_train, X_valid, y_valid)`
    (which must train the model and return (preds_train, preds_valid)), aggregates
    AUC/logloss across folds, and applies the "overfit -> score 0.5" guard shared
    by the XGBoost and CatBoost objectives.

    Parameters
    ----------
    fit_predict_fn : Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], tuple]
        Trains the model for one fold and returns (preds_train, preds_valid).
    metric : str
        Only used as a label for the printed/aggregated metric name; the score
        itself is always ROC AUC, matching the original behavior of both callers.

    Returns
    -------
    float
        Mean validation AUC across folds, or 0.5 if train/valid AUC diverge by
        more than `tol` (relative), signaling overfitting.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=random_state is not None, random_state=random_state)
    cv_results = {
        f'{metric}_train': [],
        'logloss_train': [],
        f'{metric}_valid': [],
        'logloss_valid': []
    }

    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        preds_train, preds_valid = fit_predict_fn(X_train, y_train, X_valid, y_valid)

        cv_results[f'{metric}_train'].append(roc_auc_score(y_train, preds_train))
        cv_results['logloss_train'].append(log_loss(y_train, preds_train))
        cv_results[f'{metric}_valid'].append(roc_auc_score(y_valid, preds_valid))
        cv_results['logloss_valid'].append(log_loss(y_valid, preds_valid))

        if trial is not None:  # fold-level pruning: report the running mean validation score
            trial.report(float(np.mean(cv_results[f'{metric}_valid'])), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])
    metric_diff = metric_train / metric_valid - 1

    loss_train = np.mean(cv_results['logloss_train'])
    loss_valid = np.mean(cv_results['logloss_valid'])

    if trial is None:  # inside a study the trial table already has this
        print('Finished!')
        print(f'Train {metric}: {metric_train}')
        print(f'Valid {metric}: {metric_valid}')
        print(f'Train Loss: {loss_train}')
        print(f'Valid Loss: {loss_valid}')

    if metric_diff > tol:
        metric_valid = 0.5

    return metric_valid

def optuna_optimizer_xgb_classifier(
    trial, X, y, metric='auc', tol=0.1, random_state=1000, n_splits=4,
    base_params=None, stage=None, n_estimators_max=2000, early_stopping_rounds=100
):
    """
    Optuna objective for an XGBoost classifier: mean validation AUC (to be MAXIMIZED).

    The search space comes from schema/param_grid.json ("xgb").

    Parameters
    ----------
    base_params : dict, optional
        Parameters held fixed (e.g. the best of stage 1 while tuning stage 2).
    stage : {1, 2, None}
        1 = tree structure, 2 = boosting / regularization, None = everything in one study.
    n_estimators_max, early_stopping_rounds :
        Each fold trains up to `n_estimators_max` rounds with early stopping on the validation
        fold; the mean best number of rounds is stored on the trial as user attr `n_estimators`.

    Returns
    -------
    float
        Mean validation AUC across folds (0.5 if the train/valid gap exceeds `tol`).
    """
    grid = load_param_grid("xgb")
    stages = [1, 2] if stage is None else [stage]
    params = {**_XGB_DEFAULTS, **(base_params or {})}
    for stage_ in stages:
        for name in _XGB_STAGE_PARAMS[stage_]:
            params[name] = _suggest_from_grid(trial, name, grid[name])

    param_grid = {
        **params,
        "objective": grid.get("objective", "binary:logistic"),
        "eval_metric": metric,
        "seed": random_state,
        "n_jobs": grid.get("n_jobs", -1),
    }

    best_iterations = []

    def _fit_predict(X_train, y_train, X_valid, y_valid):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        bst = xgb.train(
            param_grid, dtrain, num_boost_round=n_estimators_max,
            evals=[(dvalid, "validation")], early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_iterations.append(bst.best_iteration + 1)
        rounds = (0, bst.best_iteration + 1)
        return bst.predict(dtrain, iteration_range=rounds), bst.predict(dvalid, iteration_range=rounds)

    score = _cv_score_with_overfit_guard(_fit_predict, X, y, metric, tol, n_splits, random_state, trial=trial)
    trial.set_user_attr("n_estimators", int(np.mean(best_iterations)))
    return score

def optuna_optimizer_cat_classifier(
    trial,
    X,
    y,
    metric='AUC',
    tol=0.05,
    random_state=1000,
    n_splits=4
):
    """
    Function to optimize hyperparameter searching with OPTUNA - CatBoost.
    """

    param_dict = load_param_grid("cat")

    param_grid = {
        'n_estimators': trial.suggest_int(
            'n_estimators',
            low=param_dict["n_estimators"]["low"],
            high=param_dict["n_estimators"]["high"],
            step=param_dict["n_estimators"]["step"],
            log=param_dict["n_estimators"]["log"]
        ),
        'random_state': random_state,
        'eval_metric': metric,
        'max_depth': trial.suggest_int(
            'max_depth',
            low=param_dict["max_depth"]["low"],
            high=param_dict["max_depth"]["high"],
            step=param_dict["max_depth"]["step"],
            log=param_dict["max_depth"]["log"]
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate',
            low=param_dict["learning_rate"]["low"],
            high=param_dict["learning_rate"]["high"],
            step=param_dict["learning_rate"]["step"],
            log=param_dict["learning_rate"]["log"]
        ),
        'objective': trial.suggest_categorical(
            "objective",
            choices=param_dict["objective"]
        ),
        'colsample_bylevel': trial.suggest_float(
            "colsample_bylevel",
            low=param_dict["colsample_bylevel"]["low"],
            high=param_dict["colsample_bylevel"]["high"],
            step=param_dict["colsample_bylevel"]["step"],
            log=param_dict["colsample_bylevel"]["log"]
        ),
        'l2_leaf_reg': trial.suggest_float(
            "l2_leaf_reg",
            low=param_dict["l2_leaf_reg"]["low"],
            high=param_dict["l2_leaf_reg"]["high"],
            step=param_dict["l2_leaf_reg"]["step"],
            log=param_dict["l2_leaf_reg"]["log"]
        ),
        'boosting_type': trial.suggest_categorical(
            "boosting_type",
            choices=param_dict["boosting_type"]
        ),
        'bootstrap_type': trial.suggest_categorical(
            "bootstrap_type",
            choices=param_dict["bootstrap_type"]
        )
    }

    if param_grid["bootstrap_type"] == "Bayesian":
        param_grid["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            low=param_dict["bagging_temperature"]["low"],
            high=param_dict["bagging_temperature"]["high"]
        )
    elif param_grid["bootstrap_type"] == "Bernoulli":
        param_grid["subsample"] = trial.suggest_float(
            "subsample",
            low=param_dict["subsample"]["low"],
            high=param_dict["subsample"]["high"]
        )

    skf = StratifiedKFold(n_splits=n_splits)
    cv_results = {
        f'{metric}_train': [],
        'logloss_train': [],
        f'{metric}_valid': [],
        'logloss_valid': []
    }

    fit_params = {
        'early_stopping_rounds': 100,
        'verbose_eval': 0
    }

    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        fit_params['eval_set'] = [(X_valid, y_valid), (X_train, y_train)]

        mdl = CatBoostClassifier(**param_grid)
        mdl.fit(
            X_train,
            y_train,
            **fit_params
        )

        preds_train = mdl.predict(X_train)
        preds_valid = mdl.predict(X_valid)

        cv_results[f'{metric}_train'].append(roc_auc_score(y_train, preds_train))
        cv_results['logloss_train'].append(log_loss(y_train, preds_train))
        cv_results[f'{metric}_valid'].append(roc_auc_score(y_valid, preds_valid))
        cv_results['logloss_valid'].append(log_loss(y_valid, preds_valid))

    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])
    metric_diff = metric_train / metric_valid - 1
    loss_train = np.mean(cv_results['logloss_train'])
    loss_valid = np.mean(cv_results['logloss_valid'])

    print('Finished!')
    print(f'Train {metric}: {metric_train}')
    print(f'Valid {metric}: {metric_valid}')
    print(f'Train Loss: {loss_train}')
    print(f'Valid Loss: {loss_valid}')

    if metric_diff > tol:
        metric_valid = 0.5

    return metric_valid

def optuna_optimizer_lgb_classifier(trial, X, y, metric='auc', tol=0.1,
                                    random_state=10000, n_splits=4):
    '''
    Function to optimize hyperparameter searching with OPTUNA - LightGBM
    '''
    param_dict = load_param_grid("lgb")

    param_grid = {
        'objective': param_dict["objective"],
        'boosting_type': param_dict["boosting_type"],
        'first_metric_only': param_dict["first_metric_only"],
        'is_unbalance': param_dict["is_unbalance"],
        'random_state': random_state,
        'verbose': param_dict["verbose"],
        'n_estimators': trial.suggest_int(
            'n_estimators',
            low=param_dict["n_estimators"]["low"],
            high=param_dict["n_estimators"]["high"],
            step=param_dict["n_estimators"]["step"],
            log=param_dict["n_estimators"]["log"]
        ),
        'max_depth': trial.suggest_int(
            'max_depth',
            low=param_dict["max_depth"]["low"],
            high=param_dict["max_depth"]["high"],
            step=param_dict["max_depth"]["step"],
            log=param_dict["max_depth"]["log"]
        ),
        'num_leaves': trial.suggest_int(
            'num_leaves',
            low=param_dict["num_leaves"]["low"],
            high=param_dict["num_leaves"]["high"],
            step=param_dict["num_leaves"]["step"],
            log=param_dict["num_leaves"]["log"]
        ),
        'min_split_gain': trial.suggest_float(
            'min_split_gain',
            low=param_dict["min_split_gain"]["low"],
            high=param_dict["min_split_gain"]["high"],
            step=param_dict["min_split_gain"]["step"],
            log=param_dict["min_split_gain"]["log"]
        ),
        'max_bin': trial.suggest_int(
            'max_bin',
            low=param_dict["max_bin"]["low"],
            high=param_dict["max_bin"]["high"],
            step=param_dict["max_bin"]["step"],
            log=param_dict["max_bin"]["log"]
        ),
        'path_smooth': trial.suggest_float(
            'path_smooth',
            low=param_dict["path_smooth"]["low"],
            high=param_dict["path_smooth"]["high"],
            step=param_dict["path_smooth"]["step"],
            log=param_dict["path_smooth"]["log"]
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate',
            low=param_dict["learning_rate"]["low"],
            high=param_dict["learning_rate"]["high"],
            step=param_dict["learning_rate"]["step"],
            log=param_dict["learning_rate"]["log"]
        ),
        'lambda_l1': trial.suggest_float(
            'lambda_l1',
            low=param_dict["lambda_l1"]["low"],
            high=param_dict["lambda_l1"]["high"],
            step=param_dict["lambda_l1"]["step"],
            log=param_dict["lambda_l1"]["log"]
        ),
        'lambda_l2': trial.suggest_float(
            'lambda_l2',
            low=param_dict["lambda_l2"]["low"],
            high=param_dict["lambda_l2"]["high"],
            step=param_dict["lambda_l2"]["step"],
            log=param_dict["lambda_l2"]["log"]
        ),
        'bagging_fraction': trial.suggest_float(
            'bagging_fraction',
            low=param_dict["bagging_fraction"]["low"],
            high=param_dict["bagging_fraction"]["high"],
            step=param_dict["bagging_fraction"]["step"],
            log=param_dict["bagging_fraction"]["log"]
        ),
        'bagging_freq': trial.suggest_int(
            'bagging_freq',
            low=param_dict["bagging_freq"]["low"],
            high=param_dict["bagging_freq"]["high"],
            step=param_dict["bagging_freq"]["step"],
            log=param_dict["bagging_freq"]["log"]
        )
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(
        trial, metric=metric, valid_name='valid'
    )

    fit_params = {
        'eval_names': ['train', 'valid'],
        'eval_class_weight': ['balanced'],
        'eval_metric': metric,
        'early_stopping_rounds': 100,
        'callbacks': [pruning_callback],
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=n_splits)
    cv_results = {
        f'{metric}_train': [],
        'loss_train': [],
        f'{metric}_valid': [],
        'loss_valid': []
    }

    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        fit_params['eval_set'] = [(X_valid, y_valid), (X_train, y_train)]

        mdl = LGBMClassifier(**param_grid)
        mdl.fit(X_train, y_train, **fit_params)

        cv_results[f'{metric}_train'].append(mdl.best_score_['train'][metric])
        cv_results['loss_train'].append(mdl.best_score_['train']['binary_logloss'])
        cv_results[f'{metric}_valid'].append(mdl.best_score_['valid'][metric])
        cv_results['loss_valid'].append(mdl.best_score_['valid']['binary_logloss'])

    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])
    metric_diff = metric_train / metric_valid - 1
    loss_train = np.mean(cv_results['loss_train'])
    loss_valid = np.mean(cv_results['loss_valid'])

    print('Finished!')
    print(f'Train {metric}: {metric_train}')
    print(f'Valid {metric}: {metric_valid}')
    print(f'Train Loss: {loss_train}')
    print(f'Valid Loss: {loss_valid}')

    if metric_diff > tol:
        metric_valid = 0.5

    return metric_valid

def oot_optuna_optimizer_lgb_classifier(trial, X, y, cv, tol=0.1, random_state=42):
    """
    Function to optimize LightGBM parameters using Optuna with cross-validation, early stopping,
    and a penalty for overfitted trials to reduce variance.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X : pd.DataFrame
        Dataframe containing features.
    y : pd.DataFrame
        Dataframe containing labels.
    cv : sklearn.model_selection.KFold
        Cross-validator object.
    tol : float
        Tolerance for overfitting; trials with '(auc_train / auc_test - 1 > tol)'
        get a score of 0.5.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    objective : float
        Mean test AUC of folds.
    """

    # Create parameter grid for study
    param_grid = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "random_state": random_state,
        "verbose": -1,
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, step=20),
        "subsample_freq": 1,
        "is_unbalance": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.41, step=0.05),
        "top_rate": trial.suggest_float("top_rate", 0.01, 0.50, step=0.01),
        "other_rate": trial.suggest_float("other_rate", 0.01, 0.50, step=0.01),
        "subsample": trial.suggest_float("subsample", 0.25, 0.75, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.75, step=0.05),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 2, 16),
        "max_bin": trial.suggest_int("max_bin", 5, 15, step=1),
        "reg_alpha": trial.suggest_float("reg_alpha", 8, 11, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 8, 11, step=0.1),
        "path_smooth": trial.suggest_float("path_smooth", 8, 11, step=0.1)
    }

    min_child_proportion = trial.suggest_float("min_child_proportion", 0.005, 0.1, step=0.005)

    # Fit parameters for evaluation set and pruning
    fit_params = {
        "eval_names": ["train", "test"],
        "eval_metric": ["auc"],
        "early_stopping_rounds": 10,
        "verbose": False
    }

    # Iterate over only the upper half of splits
    cv_results = {"auc_train": [], "auc_test": []}
    for train_index, test_index in [*cv.split(X)][cv.n_splits // 2:]:
        # Get splits
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Update 'fit_params' for early stopping with splits
        fit_params["eval_set"] = [(X_train, y_train), (X_test, y_test)]

        # Initialize model with sampled parameters and fit
        param_grid["min_child_samples"] = int(len(X_train) * min_child_proportion)
        lgbm = LGBMClassifier(**param_grid)
        lgbm.fit(X_train, y_train, **fit_params)

        # Append score to cv results
        cv_results["auc_train"].append(lgbm.best_score_["train"]["auc"])
        cv_results["auc_test"].append(lgbm.best_score_["test"]["auc"])

    # Compute mean AUC for splits
    auc_train = np.mean(cv_results["auc_train"])
    auc_test = np.mean(cv_results["auc_test"])
    auc_diff = auc_train / auc_test - 1
    objective = auc_test
    if auc_diff > tol:
        objective = 0.5

    return objective