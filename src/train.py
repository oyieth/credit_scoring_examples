import sys
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
import xgboost as xgb
import catboost as cat
import lightgbm as lgbm
import optuna
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
#from sklearn import metrics
from sklearn.metrics import (
    average_precision_score, roc_curve, roc_auc_score,
    precision_recall_curve, log_loss)
from src.utils import is_jsonable, optuna_optimizer_xgb_classifier, optuna_optimizer_cat_classifier, optuna_optimizer_lgb_classifier,oot_optuna_optimizer_lgb_classifier
colors = ['#e71d36', '#fdd85d', '#6798c0', '#b0b8b4ff', '#977390']
colors_segment = {
    'RNH': '#1982c4',
    'PRIME': '#b7d1da',
    'PRNH': '#6a4c93',
    'NRNH': '#ff595e',
    'RH': '#8ac926',
    'BSTH': '#ffca3a',
    'NRH': '#f79824',
}
log_path = 'log.txt'
def get_distributions(distributions: str):
    """
    Retrieve parameter distributions for hyperparameter tuning of different models.
    Parameters:
    ----------
    distributions : str
        A string indicating the desired set of parameter distributions.
    Returns:
    -------
    dict
        A dictionary containing parameter names as keys and distribution objects
        as values, which can be used for hyperparameter optimization. The distributions
        are typically from `scipy.stats`, such as `randint` for integer ranges and
        `uniform` for continuous ranges.
    """
    valid_distributions = [
                            'xgb1',
                            'lgb1',
                            'xgb2',
                            'lgb2',
                            'xgb3',
                            'lgb3',
                            ]
    if distributions in valid_distributions:
        if distributions == 'xgb1':
        # Even Less Overfit
            param_distributions = {
            'max_depth': randint(low=2, high=3),
            'learning_rate': uniform(loc=0.00001, scale=0.1),
            'gamma': uniform(loc=1.0, scale=10.0),
            'min_child_weight': uniform(loc=7.0, scale=20.0),
            'max_delta_step': uniform(loc=0.5, scale=2),
            'subsample': uniform(loc=0.01, scale=0.3),
            'colsample_bytree': uniform(loc=0.01, scale=0.3),
            'colsample_bylevel': uniform(loc=0.5, scale=0.499),
            'reg_alpha': uniform(loc=0.001, scale=10),
            'reg_lambda': uniform(loc=0.001, scale=10),
            'n_estimators': randint(low=20, high=100),
            'objective': ["binary:logistic"]
        }

        elif distributions == 'lgb1':
        # Even Less Overfit
            param_distributions = {
            'max_depth': randint(low=2, high=3),
            'learning_rate': uniform(loc=0.00001, scale=0.1),
            'path_smooth':  uniform(loc=1.0, scale=10.0),
            "num_leaves": randint(low=7, high=20),
            'reg_alpha': uniform(loc=0.001, scale=10),
            'reg_lambda': uniform(loc=0.001, scale=10),
            "max_bin": randint(low=1000, high=5000),
            'n_estimators': randint(low=20, high=100),
            'objective': ["binary"]
        }

        elif distributions == 'xgb2':
        # Less Overfit
            param_distributions = {
            'max_depth': randint(low=2, high=4),
            'learning_rate': uniform(loc=0.001, scale=0.4),
            'gamma': uniform(loc=3.0, scale=4.0),
            'min_child_weight': uniform(loc=1, scale=10.0),
            'max_delta_step': uniform(loc=0.5, scale=2),
            'subsample': uniform(loc=0.25, scale=0.499),
            'colsample_bytree': uniform(loc=0.25, scale=0.499),
            'colsample_bylevel': uniform(loc=0.5, scale=0.499),
            'reg_alpha': uniform(loc=0.001, scale=10),
            'reg_lambda': uniform(loc=0.001, scale=10),
            'n_estimators': randint(low=20, high=1000),
            'objective': ["binary:logistic"]
        }

        elif distributions == 'lgb2':
        # Less Overfit
            param_distributions = {
            'max_depth': randint(low=2, high=4),
            'learning_rate': uniform(loc=0.001, scale=0.4),
            'path_smooth':  uniform(loc=3.0, scale=4.0),
            "num_leaves":  randint(low=2, high=10),
            'reg_alpha': uniform(loc=0.001, scale=10),
            'reg_lambda': uniform(loc=0.001, scale=10),
            "max_bin": randint(low=1000, high=5000),
            'n_estimators': randint(low=20, high=100),
            'objective': ["binary"]
        }

        elif distributions == 'xgb3':
        # Less Bias
            param_distributions = {
            'max_depth': randint(low=2, high=3),
            'learning_rate': uniform(loc=0.00001, scale=0.1),
            'gamma': uniform(loc=5.0, scale=3.0),
            'min_child_weight': uniform(loc=17.0, scale=4.0),
            'max_delta_step': uniform(loc=0.5, scale=2),
            'subsample': uniform(loc=0.01, scale=0.3),
            'colsample_bytree': uniform(loc=0.01, scale=0.3),
            'colsample_bylevel': uniform(loc=0.5, scale=0.499),
            'reg_alpha': uniform(loc=5.001, scale=1),
            'reg_lambda': uniform(loc=1.001, scale=1),
            'n_estimators': randint(low=20, high=100),
            'objective': ["binary:logistic"]
        }

        elif distributions == 'lgb3':
        # Less Bias
            param_distributions = {
            'max_depth': randint(low=2, high=6),
            'path_smooth':  uniform(loc=5.0, scale=10.0),
            "num_leaves":   randint(low=5, high=10),
            'reg_alpha':uniform(loc=3.0, scale=10.0),
            'reg_lambda': uniform(loc=3.0, scale=10.0),
            'n_estimators': randint(low=50, high=100),
            "max_bin": randint(low=10000, high=50000),
            'objective': ["binary"]
        }

    else:
        raise ValueError(f"Unknown distribution type: {distributions}. Available distribution: {','.join(item for item in valid_distributions)}")

    return param_distributions


def init_model(model_type, params=None):
    if params is not None:
        if model_type == 'xgb':
            print('Trainig XGBClassifier with best hyperparameters...')
            model = xgb.XGBClassifier(**params)

        elif model_type == 'cat':
            print('Trainig CatBoostClassifier with best hyperparameters...')
            model = cat.CatBoostClassifier(**params)

        elif model_type == 'lgb':
            print('Trainig LGBMClassifier with best hyperparameters...')
            model = lgbm.LGBMClassifier(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        if model_type == 'xgb':
            print('Trainig XGBClassifier with best hyperparameters...')
            model = xgb.XGBClassifier()

        elif model_type == 'cat':
            print('Trainig CatBoostClassifier with best hyperparameters...')
            model = cat.CatBoostClassifier()

        elif model_type == 'lgb':
            print('Trainig LGBMClassifier with best hyperparameters...')
            model =lgbm.LGBMClassifier()

        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return model

def train_with_stored_hyperparameters(x_train, y_train,
                                      r_metrics_file, model_type,
                                      write_paths):
    """
    Train a model using hyperparameters stored in a JSON file.

    Parameters:
    ----------
    x_train : array-like
        The training features.

    y_train : array-like
        The target variable for training.

    r_metrics_file : str
        The path to the JSON file containing the stored hyperparameters.

    model_type : str
        The name of the model to train ('xgb', 'cat', or 'lgb').

    Returns:
    -------
    model
        The trained model instance.
    """
    # Loading best params
    with open(r_metrics_file, encoding='utf-8') as s_file:
        stored = json.load(s_file)
    # Either a metrics file ({'params': {...}, ...}) or a plain hyperparameter dict
    params = stored['params'] if 'params' in stored else stored

    model = init_model(model_type, params)
    model = model.fit(x_train, y_train, verbose=False)

    obtain_top_features(x_train, model, write_paths.fi_path)
    print(f"Feature important saved as {write_paths.fi_path}")
    return model


def train_with_randomized_search(x_train, y_train, model_type,
                                random_state, distributions,
                                scoring, n_splits, n_iter, write_paths):
    """
    Train a model using RandomizedSearchCV with specified hyperparameter distributions.

    Parameters:
    ----------
    x_train : array-like
        The training features.

    y_train : array-like
        The target variable for training.

    model_type : str
        The name of the model to train ('xgb', 'cat', or 'lgb').

    random_state : int
        Seed for random number generation.

    distributions : str
        Identifier for the hyperparameter distributions to use.

    scoring : str
        Scoring metric for evaluating model performance.

    n_splits : int
        Number of cross-validation folds.

    n_iter : int
        Number of iterations for RandomizedSearchCV.

    write_paths : object
        Object containing paths for saving output, including feature importance.

    Returns:
    -------
    model
        The trained model instance with the best hyperparameters.
    """
    # Retrieving distribution dictionary
    param_distributions = get_distributions(distributions)

    if model_type == 'xgb':
        print('Trainig XGBClassifier with RandomizedSearchCV...')
        clf = xgb.XGBClassifier(
            random_state=random_state, use_label_encoder=False, eval_metric='logloss')

    elif model_type == 'cat':
        print('Trainig CatBoostClassifier with RandomizedSearchCV...')
        clf = cat.CatBoostClassifier(random_state=random_state)

    elif model_type == 'lgb':
        print('Trainig LGBMClassifier with RandomizedSearchCV...')
        clf = lgbm.LGBMClassifier(random_state=random_state)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Instantiating randomized search
    rsearch = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_distributions,
                random_state=random_state,
                scoring=scoring,
                cv=n_splits,
                verbose= False,
                n_iter=n_iter,
                n_jobs=-1
    )

    # Training with randomized search and fit params
    models = rsearch.fit(x_train, y_train)

    # Printing best model results and save best model and params
    print('Best score:', models.best_score_)
    print('Best hyperparameters:', models.best_params_, end='\n')
    model = models.best_estimator_

    # Feature importance
    obtain_top_features(x_train=x_train, model=model, fi_path=write_paths.fi_path)

    return model

def _validate_tuning_inputs(x_train, y_train, model_type, n_splits):
    """Fail early, with a readable message, on inputs that would crash deep inside a trial."""
    if model_type not in ('xgb', 'cat', 'lgb'):
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type != 'xgb':
        raise NotImplementedError(
            f"No search space for '{model_type}' in schema/param_grid.json yet; add a '{model_type}' "
            "section there before tuning it.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    x = x_train if isinstance(x_train, pd.DataFrame) else pd.DataFrame(x_train)
    y = y_train if isinstance(y_train, pd.Series) else pd.Series(np.asarray(y_train))
    x, y = x.reset_index(drop=True), y.reset_index(drop=True)
    if len(x) != len(y):
        raise ValueError(f"x_train has {len(x)} rows but y_train has {len(y)}")
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(f"y_train has a single class ({counts.index[0]}); nothing to tune")
    if counts.min() < n_splits:
        raise ValueError(
            f"the minority class has {counts.min()} rows, fewer than n_splits={n_splits}")
    return x, y


def create_optuna_study(x_train, y_train, model_type,
                        n_splits, random_state, timeout, n_trials, warm_start=None,
                        storage=None, study_name='tuning', sampler='tpe'):
    """
    Create and optimize an Optuna study for hyperparameter tuning.

    Maximizes the cross-validated validation AUC (see `optuna_optimizer_*`) with the
    Hyperband pruner and TPE sampler. XGBoost is tuned in two stages (tree structure, then
    boosting / regularization); `timeout` and `n_trials` apply to each stage.

    Parameters:
    ----------
    x_train, y_train : training features / target.
    model_type : 'xgb', 'cat' or 'lgb'.
    n_splits : number of cross-validation splits.
    random_state : seed for the sampler and the folds.
    timeout : time limit in seconds (per stage for xgb).
    n_trials : number of trials (per stage for xgb).
    warm_start : dict, optional
        Parameters to try first (XGBoost: names outside a stage's search space are ignored).
    storage : str, optional
        Optuna storage URL (e.g. 'sqlite:///models/model_nb/optuna.db'). With it the studies
        are persisted and re-running resumes them (`<study_name>_stage1`, `<study_name>_stage2`).
    sampler : {'tpe', 'random'}
        'random' gives the equal-budget random-search control.

    Returns:
    -------
    study : optuna.study.Study
        The (last) optimized study.
    params : dict
        Best hyperparameters; for xgb it includes `n_estimators` from early stopping.
    """
    x_train, y_train = _validate_tuning_inputs(x_train, y_train, model_type, n_splits)
    if sampler not in ('tpe', 'random'):
        raise ValueError(f"sampler must be 'tpe' or 'random', got {sampler!r}")

    def new_study(suffix=''):
        sampler_obj = (optuna.samplers.TPESampler(seed=random_state) if sampler == 'tpe'
                       else optuna.samplers.RandomSampler(seed=random_state))
        return optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(), direction='maximize', sampler=sampler_obj,
            storage=storage, study_name=f'{study_name}{suffix}' if storage else None,
            load_if_exists=True)

    study = new_study('_stage0')

    if model_type == 'xgb':
        from src.utils import _XGB_STAGE_PARAMS

        def run_stage(stage, base_params):
            stage_study = new_study(f'_stage{stage}')
            if warm_start:
                stage_study.enqueue_trial(
                    {k: v for k, v in warm_start.items() if k in _XGB_STAGE_PARAMS[stage]})
            stage_study.optimize(
                lambda trial: optuna_optimizer_xgb_classifier(
                    trial, X=x_train, y=y_train, n_splits=n_splits, random_state=random_state,
                    base_params=base_params, stage=stage),
                timeout=timeout, n_trials=n_trials)
            return stage_study

        study_1 = run_stage(1, base_params={})
        study = run_stage(2, base_params=study_1.best_params)
        params = {**study_1.best_params, **study.best_params}
        params['n_estimators'] = study.best_trial.user_attrs['n_estimators']
    elif model_type == 'cat':
        obj_function = lambda trial: optuna_optimizer_cat_classifier(
            trial, X=x_train, y=y_train, n_splits=n_splits, random_state=random_state)
        study.optimize(obj_function, timeout=timeout, n_trials=n_trials)
        params = study.best_params
    elif model_type == 'lgb':
        obj_function = lambda trial: optuna_optimizer_lgb_classifier(
            trial, X=x_train, y=y_train, n_splits=n_splits, random_state=random_state)
        study.optimize(obj_function, timeout=timeout, n_trials=n_trials)
        params = study.best_params
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    params['random_state'] = random_state
    return study, params


def create_oot_optuna_study(x_train, y_train, model_type,
                        n_splits, random_state, timeout, n_trials):
    """
     Create and optimize an Optuna study for hyperparameter tuning of LightGBM.

    This function sets up an Optuna study specifically for optimizing
    hyperparameters of the LightGBM classifier using time series cross-validation.

    Parameters:
    ----------
    x_train : array-like
        Training feature data.

    y_train : array-like
        Training target data.

    model_type : str
        The name of the model to optimize ('lgb' for LightGBM).

    n_splits : int
        Number of splits for time series cross-validation.

    random_state : int
        Seed for reproducibility.

    timeout : int
        Time limit for optimization in seconds.

    n_trials : int
        Number of optimization trials.

    Returns:
    -------
    study : optuna.study.Study
        The optimized Optuna study object.

    params : dict
        The best hyperparameters found during optimization.
    """
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')

    if model_type == 'lgb':
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # pylint: disable=unnecessary-lambda-assignment
        obj_function = lambda trial: oot_optuna_optimizer_lgb_classifier(
            trial, X=x_train, y=y_train, cv=tscv, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    study.optimize(obj_function, timeout=timeout, n_trials=n_trials)

    # Retrieving best params dict
    params = study.best_params
    params['random_state'] = random_state

    return study, params


def train_with_optuna(
    x_train, y_train, model_type,random_state, n_splits,
    w_suffix, write_paths, timeout, n_trials
):
    """
    Train a model using Optuna for hyperparameter optimization.

    This function sets up an Optuna study to find the best hyperparameters
    for a specified machine learning model (XGBoost, CatBoost, or LightGBM),
    trains the model with those parameters, and saves the model's performance metrics.

    Parameters:
    ----------
    x_train : array-like
        Training feature data.

    y_train : array-like
        Training target data.

    model_type : str
        The name of the model to train ('xgb', 'cat', or 'lgb').

    random_state : int
        Seed for reproducibility.

    n_splits : int
        Number of cross-validation splits.

    w_suffix : str
        Suffix for saving plots.

    write_paths : object
        Object containing paths for saving results.

    timeout : int
        Time limit for Optuna optimization in seconds.

    n_trials : int
        Number of trials for Optuna optimization.

    Returns:
    -------
    model : object
        The trained machine learning model.
    """
    if model_type == 'xgb':
        print('Trainig XGBClassifier with Optuna...')
        # Creating optuna study
        study, params = create_optuna_study(
            x_train=x_train, y_train=y_train, model_type=model_type,
            n_splits=n_splits, random_state=random_state, timeout=timeout, n_trials=n_trials
        )
        # Instantiating with best params
        model = xgb.XGBClassifier(**params)

    elif model_type == 'cat':
        print('Trainig CatBoostClassifier with Optuna...')

        # Creating optuna study
        study, params = create_optuna_study(
            x_train=x_train, y_train=y_train, model_type=model_type,
            n_splits=n_splits, random_state=random_state, timeout=timeout, n_trials=n_trials
        )
        # Instantiating with best params
        model = cat.CatBoostClassifier(**params)


    elif model_type == 'lgb':
        print('Trainig LGBMClassifier with Optuna...')
        # Creating optuna study
        study, params = create_optuna_study(
            x_train=x_train, y_train=y_train, model_type=model_type,
            n_splits=n_splits, random_state=random_state, timeout=timeout, n_trials=n_trials
        )
        # Retrieving best params dict
        params['objective'] = 'binary'
        params['boosting_type'] = 'gbdt'
        params['is_unbalance'] = True
        params['first_metric_only'] = True
        # Instantiating with best params
        model = lgbm.LGBMClassifier(**params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Getting hyperparameter impurity importances
    try:
        param_importance = optuna.importance.get_param_importances(study)
        param_importance_df = pd.DataFrame(
            index=param_importance.keys(), data=param_importance.values(),
            columns=['impurity_importance'])
        param_importance_df.plot(kind='barh')
        plt.title(f'{w_suffix} Impurity Importances')
        plt.savefig(write_paths.imp_path, dpi=500, bbox_inches='tight')
        plt.close()
    except: # pylint: disable=W0702
        pass

    # Training with randomized search and fit params
    model.fit(x_train, y_train, verbose=True)

    # Printing best model results and save best model and params
    print('Best score:', model.score(x_train, y_train))
    print('Best hyperparameters:', params, end='\n')

    #feature importances
    obtain_top_features(x_train, model, write_paths.fi_path)

    return model


def train_with_oot_optuna(
    x_train, y_train, model_type, random_state, n_splits,
    w_suffix, write_paths, timeout, n_trials
):
    """
    Train a model using out-of-time validation with Optuna for hyperparameter optimization.

    This function sets up an Optuna study to find the best hyperparameters for
    a LightGBM classifier, trains the model with those parameters, and saves
    the model's performance metrics.

    Parameters:
    ----------
    x_train : array-like
        Training feature data.

    y_train : array-like
        Training target data.

    model_type : str
        The name of the model to train (currently only supports 'lgb').

    random_state : int
        Seed for reproducibility.

    n_splits : int
        Number of cross-validation splits.

    w_suffix : str
        Suffix for saving plots.

    write_paths : object
        Object containing paths for saving results.

    timeout : int
        Time limit for Optuna optimization in seconds.

    n_trials : int
        Number of trials for Optuna optimization.

    Returns:
    -------
    model : object
        The trained LightGBM model.
    """
    if model_type == 'lgb':
        print('Trainig LGBMClassifier with Optuna...')
        # Creating optuna study
        study, params = create_oot_optuna_study(
            x_train=x_train, y_train=y_train, model_type=model_type,
            n_splits=n_splits, random_state=random_state, timeout=timeout, n_trials=n_trials
        )

        # Retrieving best params dict
        params['objective'] = 'binary'
        params['boosting_type'] = 'gbdt'
        params['is_unbalance'] = True
        params['first_metric_only'] = True
        # Instantiating with best params
        model = lgbm.LGBMClassifier(**params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Getting hyperparameter impurity importances
    try:

        param_importance = optuna.importance.get_param_importances(study)
        param_importance_df = pd.DataFrame(
            index=param_importance.keys(), data=param_importance.values(),
            columns=['impurity_importance'])
        param_importance_df.plot(kind='barh')
        plt.title(f'{w_suffix} Impurity Importances')
        plt.savefig(write_paths.imp_path, dpi=500, bbox_inches='tight')
        plt.close()

    except: # pylint: disable=W0702
        pass

    # Training with randomized search and fit params
    model.fit(x_train, y_train, verbose=True)

    # Printing best model results and save best model and params
    print('Best score:', model.score(x_train, y_train))
    print('Best hyperparameters:', params, end='\n')

    #feature importances
    obtain_top_features(x_train, model, write_paths.fi_path)

    return model

def obtain_top_features(x_train, model, fi_path):
    """
    Extract and plot the top feature importances from a trained model.

    This function retrieves the feature importances from a given trained model,
    handles various model types, and visualizes the top feature importances.
    The resulting plot is saved to the specified file path.

    Parameters:
    ----------
    x_train : DataFrame
        The training dataset used for model fitting. It is utilized to retrieve feature names.

    model : object
        The trained model from which feature importances are extracted. It can be of various types,
        such as calibrated classifiers or tree-based models.

    fi_path : str
        The file path where the feature importance plot will be saved.

    Returns:
    -------
    top_features : DataFrame
        A DataFrame containing the feature names and their corresponding importance scores,
        sorted in descending order of importance.
    """
    top_features = pd.DataFrame()

    # Handle different model types and extract the base estimator if available.
    try:
        model_ = model.calibrated_classifiers_[0].base_estimator
    except AttributeError:
        model_ = model

    # Attempt to get feature names from the model.
    try:
        top_features['columns'] = model_[0].get_feature_names(input_features=x_train.columns)
    except (AttributeError, IndexError, TypeError):
        try:
            top_features['columns'] = x_train.columns
        except (AttributeError, IndexError):
            top_features['columns'] = model_.booster_.feature_name()

    # Attempt to get feature importances from the model.
    try:
        top_features['importances'] = model_.feature_importances_
    except AttributeError:
        try:
            top_features['importances'] = model.best_estimator_.feature_importances_
        except AttributeError:
            try:
                top_features['importances'] = model_.get_feature_importance()
            except AttributeError:
                top_features['importances'] = model_.booster_.feature_importance(
                    importance_type='gain'
                )

    # Set the columns as the index and sort by importances.
    top_features.set_index('columns', inplace=True)
    top_features.sort_values(by='importances', ascending=True, inplace=True)

    # Plot and save the top 15 feature importances.
    top_features['importances'].tail(10).plot(kind='barh')
    plt.title('Feature Importances')
    plt.savefig(fi_path, dpi=500, bbox_inches='tight')
    plt.close()

    print(top_features.tail(10), end='\n\n')

    return top_features

def calibration_plots(x_test, y_test, original_model, calibrated_model, calibration_curves_path):
    """
     Generate and save calibration plots for comparing predicted probabilities
    of the original and calibrated models against actual observed outcomes.

    This function calculates predicted probabilities from both models, bins
    them, and plots the mean predicted probabilities against the fraction of
    positive observations.

    Parameters:
    ----------
    x_test : array-like
        Feature data for testing the models.

    y_test : array-like
        Actual target values for the test data.

    original_model : object
        The original model used for generating predictions.

    calibrated_model : object
        The calibrated model used for generating predictions.

    calibration_curves_path : str
        File path to save the calibration curve plot.

    Returns:
    -------
    None
    """
    y_scores_test = calibrated_model.predict_proba(x_test)[:, 1]
    # Creating calibration plots
    calibration_bins = np.linspace(0, 1, num=11)
    bins = pd.DataFrame(
        pd.cut(original_model.predict_proba(x_test)[:, 1],
        bins=calibration_bins), columns=['SCORE_BIN'])
    bins['PROBA_BIN'] = pd.cut(y_scores_test, bins=calibration_bins)
    bins['SCORE'] = original_model.predict_proba(x_test)[:, 1]
    bins['PROBA'] = y_scores_test
    bins['TARGET'] = y_test.reset_index().iloc[:, 1]
    score_results = pd.pivot_table(
        bins, index='SCORE_BIN', values=['TARGET', 'SCORE'], aggfunc=np.mean)
    proba_results = pd.pivot_table(
        bins, index='PROBA_BIN', values=['TARGET', 'PROBA'], aggfunc=np.mean)

    # Saving Calibration Curves
    plt.figure(figsize=(6, 6))
    plt.title('Calibration Curves')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted probability')
    plt.ylim((-0.05, 1.05))
    plt.xlim((-0.05, 1.05))
    plt.plot([0, 1], [0, 1], label='Perfectly calibrated', color='darkgray', linestyle='dotted')
    plt.plot(score_results.SCORE, score_results.TARGET, label='SCORE', color='orange', marker='s')
    plt.plot(proba_results.PROBA, proba_results.TARGET, label='PROBA', color='green', marker='s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(calibration_curves_path, dpi=500, bbox_inches='tight')
    plt.close()


def save_predicted_scores(model, x_test, test_data,
                          x_train, train_data, write_paths):
    """
    Save the predicted scores for both training and testing datasets.

    This function computes the predicted probabilities for the given
    test and train datasets, transforms these scores, and saves them
    to CSV files.

    Parameters:
    ----------
    model : object
        The trained model used for making predictions.

    x_test : array-like
        Feature data for the test set.

    test_data : DataFrame
        Original test data containing identifiers or other relevant info.

    x_train : array-like
        Feature data for the training set.

    train_data : DataFrame
        Original training data containing identifiers or other relevant info.

    write_paths : object
        Object containing file paths for saving the scored datasets.

    Returns:
    -------
    None
    """
    y_scores_test = model.predict_proba(x_test)[:, 1]
    test_data['SCORE'] = y_scores_test

    y_scores_train = model.predict_proba(x_train)[:, 1]
    train_data['SCORE'] = y_scores_train

    train_data.to_csv(write_paths.train_scored_path)
    test_data.to_csv(write_paths.test_scored_path)


def plot_roc(labels, prediction_scores, legend, color):
    """
    Function to plot the ROC curve.

    Parameters:
    -----------
    labels : array-like
        True binary labels.
    prediction_scores : array-like
        Predicted scores or probabilities.
    legend : str
        Label for the ROC curve in the plot legend.
    color : str
        Color for the ROC curve.
    """
    fpr, tpr, _ = roc_curve(labels, prediction_scores, pos_label=1)
    auc = roc_auc_score(labels, prediction_scores)
    legend_string = f"{legend} ($AUC = {auc:0.4f}$)"
    plt.plot(fpr, tpr, label=legend_string, color=color)


def plot_prc(labels, prediction_scores, legend, color):
    """
    Function to plot the Precision-Recall Curve (PRC).

    Parameters:
    -----------
    labels : array-like
        True binary labels.
    prediction_scores : array-like
        Predicted scores or probabilities.
    legend : str
        Label for the PRC curve in the plot legend.
    color : str
        Color for the PRC curve.
    """
    precision, recall, _ = precision_recall_curve(labels, prediction_scores)
    average_precision = average_precision_score(labels, prediction_scores)
    legend_string = f"{legend} ($AP = {average_precision:0.4f}$)"
    plt.plot(recall, precision, label=legend_string, color=color)


def plot_ks(labels, prediction_scores, color):
    """
    Function to plot the KS plot.

    Parameters:
    -----------
    labels : array-like
        True binary labels.
    prediction_scores : array-like
        Predicted scores or probabilities.
    color : list
        List of colors to be used for the FNR and TNR curves.

    Returns:
    --------
    ks : float
        The maximum KS statistic.
    t_ : float
        The threshold at which the maximum KS statistic occurs.
    """
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1

    plt.plot(thresholds, fnr, label='FNR (Class 1 Cum. Dist.)', color=color[0], lw=1.5)
    plt.plot(thresholds, tnr, label='TNR (Class 0 Cum. Dist.)', color=color[1], lw=1.5)

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]

    plt.vlines(
        t_,
        tnr[np.argmax(np.abs(kss))],
        fnr[np.argmax(np.abs(kss))],
        colors='red',
        linestyles='dashed'
    )

    return ks, t_


def plot_ks_2(labels, prediction_scores, legend, color):
    '''
    Function to plot KS plot
    '''
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]
    legend_string = f'{legend} ($KS = {ks:0.4f}$; $x = {t_:0.4f}$)'
    plt.plot(thresholds, kss, label=legend_string, color=color, lw=1.5)
    plt.vlines(t_, ks, 0, colors=color, linestyles='dashed', alpha=0.4)

    return ks, t_


def format_plot(title, xlabel, ylabel):
    """
    Function to format a plot.

    Parameters:
    -----------
    title : str
        The title of the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()




def get_metrics(y_real, y_scores, set_name: str, ts_list: list, scores_path: str):
    """
    Calculate and display AUC and other metrics for given scores and labels.

    Parameters:
    ----------
    y_real : array-like
        True binary labels.
    y_scores : array-like
        Predicted scores or probabilities.
    set_name : str
        Name of the dataset or set for display purposes.
    ts_list : list
        List of thresholds to calculate metrics.
    scores_path : str
        Path to save the score distribution plot.

    Returns:
    -------
    tuple
        AUC score, list of thresholds, F1 scores, precision values, and recall values.
    """
    print(f'{set_name} Results:\n')
    roc_auc = roc_auc_score(y_real, y_scores)
    print(f'    AUC: {roc_auc:.4f}\n')

    # Calculate bin distribution
    try:
        bin_distribution = pd.DataFrame(
            pd.cut(y_scores, bins=np.arange(0, 1.1, 0.1)).value_counts(),

        )
        bin_distribution.columns = ['COUNT']
    except ValueError:
        bin_distribution = pd.DataFrame(
            pd.cut(y_scores, bins=np.arange(0, 1.1, 0.1)).value_counts()
        )
        bin_distribution = bin_distribution.rename(columns={'SCORE': 'COUNT'})

    bin_distribution['Class 1'] = pd.cut(
        y_scores[y_real == True], bins=np.arange(0, 1.1, 0.1)
    ).value_counts()
    bin_distribution['Class 1'] /= bin_distribution['COUNT']
    bin_distribution['Class 0'] = pd.cut(
        y_scores[y_real == False], bins=np.arange(0, 1.1, 0.1)
    ).value_counts()
    bin_distribution['Class 0'] /= bin_distribution['COUNT']
    bin_distribution['PROPORTION'] = bin_distribution['COUNT'] / y_scores.shape[0]
    bin_distribution = bin_distribution.sort_index(ascending=True)

    plot_bin_distribution(bin_distribution, set_name, scores_path)

    f1_list, precision_list, recall_list = list_of_metrics_for_threshold(
        ts_list, y_real, y_scores)

    return roc_auc, ts_list, f1_list, precision_list, recall_list


def performance_metrics(x_train, y_train, x_test, y_test, model,
                        write_paths, manual_threshold,
                        x_test_, y_test_):
    """
    Calculate and plot the performance metrics of a model on training and testing datasets.

    This function evaluates the model's performance by computing AUC, precision, recall,
    and F1 scores, generating ROC and Precision-Recall curves, and saving the results
    to specified file paths. It also captures and returns key performance metrics in a
    dictionary.

    Parameters:
    ----------
    x_train : array-like
        Feature data for the training set.

    y_train : array-like
        Target data for the training set.

    x_test : array-like
        Feature data for the test set.

    y_test : array-like
        Target data for the test set.

    model : object
        The trained model used for making predictions.

    write_paths : object
        Object containing file paths for saving the performance metrics plots.

    manual_threshold : float
        A manually specified threshold for classification.

    x_test_ : array-like
        Additional feature data for a more extensive test set.

    y_test_ : array-like
        Additional target data for the more extensive test set.

    Returns:
    -------
    dict
        A dictionary containing various performance metrics, including
        ROC AUC, F1 score, precision, recall, thresholds, and model parameters.
    """
    y_test_ = y_test_.astype(int)

    # Getting predicted scores
    y_scores_test = model.predict_proba(x_test)[:, 1]
    y_scores_train = model.predict_proba(x_train)[:, 1]
    y_scores_test_ = model.predict_proba(x_test_)[:, 1]

    # Getting ROC AUC
    auc_test = roc_auc_score(y_test, y_scores_test)
    auc_train = roc_auc_score(y_train, y_scores_train)
    print(f'AUC\n{14 * "-"}\nTrain:\t{auc_train:.2%}\nTest:\t{auc_test:.2%}\n{14 * "-"}\n')

    # Train Results
    threshold = 0.01
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_train)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 0.00000000000000001)
    threshold = float(thresholds[np.argmax(f1_scores)])
    get_metrics(
        y_train, y_scores_train, 'Train', [threshold, manual_threshold],
        write_paths.scores_path)

    # Test Results
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores_test)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 0.00000000000000001)
    roc_auc, ts_list, f1_list, precision_list, recall_list = get_metrics(
        y_test, y_scores_test, 'Test', [threshold, manual_threshold],
        write_paths.scores_path)

    # Printing ROC curves
    plt.figure(figsize=(6, 6))
    plot_roc(y_test, y_scores_test, 'Test', colors[4])
    plot_roc(y_train, y_scores_train, 'Train', colors[2])
    format_plot(
        'ROC ' + write_paths.roc_path.name.upper(),
        'False Positive Rate', 'True Positive Rate (Recall)')
    plt.legend(prop={'size': 5})
    plt.savefig(write_paths.roc_path, dpi=500, bbox_inches='tight')
    plt.close()

    # Printing PRC curves
    plt.figure(figsize=(6, 6))
    plot_prc(y_test, y_scores_test, 'Test', colors[4])
    plot_prc(y_train, y_scores_train, 'Train', colors[2])
    format_plot('PRC ' + write_paths.prc_path.name.upper(), 'Recall', 'Precision')
    plt.legend(prop={'size': 5})
    plt.savefig(write_paths.prc_path, dpi=500, bbox_inches='tight');
    plt.close()

    # Printing KS curves
    plt.figure(figsize=(6, 6))
    plot_ks(y_test, y_scores_test, ['#fb6376', '#3a7ca5'])
    plot_ks_2(y_test, y_scores_test, 'Test', colors[4])
    format_plot('KS', 'Threshold', 'Rate (Cumulative Distribution) / KS value')
    plt.legend(prop={'size': 5})
    plt.savefig(write_paths.ks_path, dpi=500, bbox_inches='tight');
    plt.close()

    # Saving metrics
    try:
        metric_scores = {
            'roc_auc': roc_auc,
            'threshold': ts_list[0],
            'f1_score': f1_list[0],
            'precision': precision_list[0],
            'recall': recall_list[0],
            'manual_threshold': ts_list[1],
            'mt_f1_score': f1_list[1],
            'mt_precision': precision_list[1],
            'mt_recall': recall_list[1],
            'params': {k: v for k, v in model.__dict__.items() if is_jsonable(v) and k[:1] != "_"}
        }
    except:  # pylint: disable=W0702
        metric_scores = {
            'roc_auc': roc_auc,
            'threshold': ts_list[0],
            'f1_score': f1_list[0],
            'precision': precision_list[0],
            'recall': recall_list[0],
            'params': {k: v for k, v in model.__dict__.items() if is_jsonable(v) and k[:1] != "_"}
        }

    return metric_scores


def plot_bin_distribution(bin_distribution, set_name, scores_path):
    """
    Plot the bin distribution of scores and save the figure.
    Parameters:
    ----------
    bin_distribution : pandas.DataFrame
        A DataFrame containing the distribution data with at least the columns
        'Class 1', 'Class 0', and 'COUNT'. The 'Class 1' and 'Class 0' columns
        represent the counts for each class in each bin, while 'COUNT' is used
        for labeling.

    set_name : str
        The name of the dataset being visualized (e.g., "train", "test"). This
        is used for the title of the plot and the file name.

    scores_path : pathlib.Path
        Base path of the plot (e.g. models/<name>/img/scores_<name>.png). The set name is
        appended to the file stem, so train and test are saved as separate files
        (scores_<name>_train.png, scores_<name>_test.png).

    Returns:
    -------
    None
        The function saves the plot as a PNG file and does not return any value.

    """
    bin_distribution[['Class 1', 'Class 0']].plot(
        kind='barh', stacked=True, color=['#fb6376', '#3a7ca5'],
        title=f'{set_name} score distribution'
    )
    plt.legend(loc='lower right')
    print(bin_distribution.head(10))
    for i, (_, row) in enumerate(bin_distribution.iterrows()):
        plt.text(0.05, i - 0.11, str(int(row.COUNT)), color='white')
    plt.subplots_adjust(left=0.2)
    plt.savefig(scores_path.with_name(f'{scores_path.stem}_{set_name.lower()}{scores_path.suffix}'))
    plt.close()

def list_of_metrics_for_threshold(ts_list, y_real, y_scores):
    """
    Calculate precision, recall, and F1 score for a list of thresholds.

    Parameters:
    ----------
    ts_list : list
        List of threshold values to evaluate.
    y_real : array-like
        True binary labels.
    y_scores : array-like
        Predicted scores or probabilities.

    Returns:
    -------
    tuple
        Lists of F1 scores, precision, and recall for each threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_real, y_scores)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-15)
    f1_list, precision_list, recall_list = [], [], []

    for ts in ts_list:
        print(f'    For {ts:.4f} Threshold:\n')
        (
            precision, recall, f1_score
        ) = calculate_metrics(ts, precisions, recalls, f1_scores, thresholds)

        print(
            f'        Precision:   {precision:.4f} | '
            f'Recall:      {recall:.4f} | '
            f'F1 Score:    {f1_score:.4f}'
        )
        f1_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)

    return f1_list, precision_list, recall_list

def calculate_metrics(ts, precisions, recalls, f1_scores, thresholds):
    """
    Retrieve precision, recall, and F1 score at a given threshold.

    Parameters:
    ----------
    ts : float
        The threshold value to use.
    precisions, recalls, f1_scores : numpy.ndarray
        Arrays of precision, recall, and F1 score values.
    thresholds : numpy.ndarray
        Array of threshold values.

    Returns:
    -------
    tuple
        The precision, recall, and F1 score at the specified threshold.
    """
    precision = precisions[np.argwhere(thresholds >= ts)[0]][0]
    recall = recalls[np.argwhere(thresholds >= ts)[0]][0]
    f1_score = f1_scores[np.argwhere(thresholds >= ts)[0]][0]

    return precision, recall, f1_score

def train_model(
        x_train,
        y_train,
        model_type,
        read_paths,
        use_best_params,
        param_search,
        random_state,
        distributions,
        scoring,
        n_splits,
        n_iter,
        w_suffix,
        write_paths,
        timeout,
        n_trials
):
    """
    Train a machine learning model using specified parameters or conduct hyperparameter search.

    Depending on the inputs, this function either trains the model using stored hyperparameters
    or performs a hyperparameter search using various methods.

    Parameters:
    ----------
    x_train : pd.DataFrame
        The training feature set.

    y_train : pd.Series
        The target variable corresponding to the training feature set.

    model_type : str
        The name of the model to be trained.

    read_paths : object
        An object containing paths for reading metrics or other necessary files.

    use_best_params : bool
        If True, the model will be trained using the best stored hyperparameters.
        If False, hyperparameter tuning will be performed.

    param_search : str
        The method for hyperparameter search. Options include:
        - 'randomized_search': Uses randomized search for hyperparameter tuning.
        - 'optuna': Uses the Optuna framework for hyperparameter optimization.
        - 'oot_optuna': Performs out-of-time Optuna hyperparameter tuning.

    random_state : int
        Random seed for reproducibility of results.

    distributions : dict
        The distributions of hyperparameters for tuning.

    scoring : str
        The scoring metric used to evaluate the model during training.

    n_splits : int
        The number of splits for cross-validation.

    n_iter : int
        The number of iterations for randomized search.

    w_suffix : str
        A suffix for output files to differentiate runs.

    write_paths : object
        An object containing paths for saving the results of model training.

    timeout : int
        The maximum time (in seconds) to allow for the hyperparameter search.

    n_trials : int
        The number of trials to run during hyperparameter optimization.

    Returns:
    -------
    model
        The trained model object.
    """

    if use_best_params:
        if Path(read_paths.metrics_file).is_file():
            return train_with_stored_hyperparameters(
            x_train, y_train, read_paths.metrics_file, model_type,
            write_paths
            )
        else:
            raise ValueError(f"File {read_paths.metrics_file} does not exist.")

    if param_search == 'randomized_search':
        return train_with_randomized_search(
            x_train, y_train, model_type, random_state,
            distributions, scoring, n_splits, n_iter, write_paths
        )

    if param_search == 'optuna':
        return train_with_optuna(
            x_train, y_train, model_type, random_state,
            n_splits, w_suffix, write_paths,
            timeout, n_trials
        )

    if param_search == 'oot_optuna':
        return train_with_oot_optuna(
            x_train, y_train, model_type, random_state, n_splits,
            w_suffix, write_paths, timeout, n_trials
        )

    raise ValueError(f"Unknown param_search method: {param_search}")
