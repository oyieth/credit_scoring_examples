
import os
from pathlib import Path
#Define paths
current_script_dir = Path(__file__).resolve().parent #location should de in src
ROOT_DIR = current_script_dir.parent #repo location
SCHEMA_DIR = ROOT_DIR / "schema"
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
LOG_PATH = ROOT_DIR / "log.txt"

# Naming convention
#   notebooks -> each one states its own light model (model_nb) and sample size
#   train/*.py -> the full pipeline on all the data
PIPELINE_MODEL = "model_00"

class ReadWritePaths:
    """
    A class to manage file paths for saving data and related metrics.
    """
    def __init__(self, data_dict):
        self.__dict__.update(data_dict)


class ExtractPaths(ReadWritePaths):
    """
    Paths for step 1 (feature extraction): raw Kaggle tables in, `*_all_features.csv` out.
    """

    def __init__(self):
        paths = {
            'application_train_path': DATA_DIR / 'application_train.csv',
            'application_test_path': DATA_DIR / 'application_test.csv',
            'previous_application_path': DATA_DIR / 'previous_application.csv',
            'installments_payments_path': DATA_DIR / 'installments_payments.csv',
            'pos_cash_balance_path': DATA_DIR / 'POS_CASH_balance.csv',
            'credit_card_balance_path': DATA_DIR / 'credit_card_balance.csv',
            'bureau_path': DATA_DIR / 'bureau.csv',
            'bureau_balance_path': DATA_DIR / 'bureau_balance.csv',
            'train_all_features_path': DATA_DIR / 'application_train_all_features.csv',
            'test_all_features_path': DATA_DIR / 'application_test_all_features.csv',
        }
        self.__dict__.update(paths)


class WritePathsPreprocess(ReadWritePaths):
    """
    A class to manage file paths for saving preprocessed data and related metrics.
    """

    def __init__(self, name: str):
        # Make sure paths exist
        DIR = MODEL_DIR / f"{name}"
        Path(DIR).mkdir(parents=True, exist_ok=True)
        for item in ["data", "metrics", "artifacts", "schema"]:
            Path(DIR / item).mkdir(parents=True, exist_ok=True)
        paths = {
            "model_path": DIR,
            "train_dataset_path": DIR / 'data' / f'train_{name}_preprocessed.csv',
            "test_dataset_path": DIR / 'data' / f'test_{name}_preprocessed.csv',
            "completeness_report_path": DIR / 'metrics' / f'features_completeness_report{name}.csv',
            "preprocessor_path": DIR / 'artifacts' / f'preprocessor_{name}.joblib.dat',
            "ordinal_encoder_dict_path": DIR / 'schema' / f'ordinal_encoder_dict{name}.json',
        }
        self.__dict__.update(paths)


# pylint: disable=R0903
class ReadPathsPreprocess(ReadWritePaths):
    """
    A class to manage file paths for reading preprocessor
    """

    def __init__(self, name: str):
        DIR = MODEL_DIR / f"{name}"
        Path(DIR).mkdir(parents=True, exist_ok=True)
        for item in ["artifacts", ]:
            Path(DIR / item).mkdir(parents=True, exist_ok=True)
        paths = {
            "preprocessor_path": DIR / 'artifacts' / f'preprocessor_{name}.joblib.dat',
            "train_dataset_path": DIR / 'data' / f'train_{name}_preprocessed.csv',
            "test_dataset_path": DIR / 'data' / f'test_{name}_preprocessed.csv',
        }
        self.__dict__.update(paths)


class ReadPathsTrain(ReadWritePaths):
    """
    A class to handle the generation of file paths for reading metrics and model components.
    """
    def __init__(self, name: str):
        # Make sure paths exist
        DIR = MODEL_DIR / f"{name}"
        Path(DIR).mkdir(parents=True, exist_ok=True)
        for item in ["data", "metrics", "artifacts", "schema"]:
            Path(DIR / item).mkdir(parents=True, exist_ok=True)
        paths = {
        'metrics_file' : DIR / 'metrics'/f'metrics{name}.json',
        'preprocessor_path' : DIR / 'artifacts' / f'preprocessor_{name}.joblib.dat',
        'selected_features_file' : DIR / 'metrics'/f'selected_features_{name}.json',
        'model_path': DIR / "artifacts" / f'model{name}.joblib.dat',
        }
        self.__dict__.update(paths)
class WritePathsTrain:
    """
    A class to handle the generation of file paths for writing metrics and model components.
    """
    def __init__(self, name: str):
        # Make sure paths exist
        DIR = MODEL_DIR / f"{name}"
        Path(DIR).mkdir(parents=True, exist_ok=True)
        for item in ["data", "metrics", "artifacts", "schema",'img']:
            Path(DIR / item).mkdir(parents=True, exist_ok=True)
        paths = {
            'metrics_file' : DIR / 'metrics'/f'metrics{name}.json',
            'model_path' : DIR / "artifacts"/f'model{name}.joblib.dat',
            'model_pipeline_path' : DIR / "artifacts"/f'pipeline{name}.joblib.dat',
            'scores_path' : DIR / 'img'/f'scores_{name}.png',
            'roc_path' : DIR / 'img'/f'roc{name}.png',
            'prc_path' : DIR / 'img'/f'prc{name}.png',
            'prc_path_bc_score' : DIR / 'img'/f'bc_score_prc{name}.png',
            'ks_path' : DIR / 'img'/f'ks{name}.png',
            'ks_path_bc_score' : DIR / 'img'/f'bc_score_ks{name}.png',
            'calibration_curves_path' : DIR / 'img'/f'calibration_curves{name}.png',
            'shap_path' : DIR / 'img'/f'shap{name}.png',
            'shap_buro_path' : DIR / 'img'/f'shap_buro{name}.png',
            'shap_rappi_path' : DIR / 'img'/f'shap_rappi{name}.png',
            'shap_interaction_path' : DIR / 'img'/f'shap_interaction{name}.png',
            'shap_descifra_path' : DIR / 'img'/f'shap_descifra{name}.png',
            'shap_trully_path' : DIR / 'img'/f'shap_trully{name}.png',
            'explainer_path' : DIR / "artifacts"/f'explainer{name}.joblib.dat',
            'feature_scores_path' :DIR /  'metrics'/f'feature_scores{name}.csv',
            'imp_path' : DIR / 'img'/f'impurity_importances{name}.png',
            'fi_path' : DIR / 'img'/f'feature_importances{name}.png',
            'mse' : DIR / 'img'/f'mse{name}.png',
            'prediction' :DIR /  'img'/f'prediction{name}.png',
            'hist_error' : DIR / 'img'/f'hist_error{name}.png',
            'error' : DIR / 'img'/f'error{name}.png',
            'imp_path_opt1' : DIR / 'img'/f'confidence_plot_edf{name}.png',
            'imp_path_opt2' : DIR / 'img'/f'confidence_plot_optimization_history{name}.png',

        'ordinal_encoder_path' :  DIR / 'schema' / f'ordinal_encoder_dict{name}.json',
        'test_scored_path' : DIR / 'data' / f'test_scored_{name}.csv',
        'train_scored_path' : DIR / 'data' / f'train_scored_{name}.csv'
        }
        self.__dict__.update(paths)


class WritePathsFeatureSelection(ReadWritePaths):
    """
    A class to manage file paths for the feature selector pipeline (notebook 3.22).
    """
    # Metric labels the forward selection plots (see METRICS in notebook 3.22)
    CV_METRIC_LABELS = ("auc", "logloss", "ap")

    def __init__(self, name: str):
        # Make sure paths exist
        DIR = MODEL_DIR / f"{name}"
        Path(DIR).mkdir(parents=True, exist_ok=True)
        for item in ["metrics", "schema", "img"]:
            Path(DIR / item).mkdir(parents=True, exist_ok=True)
        paths = {
            'data_schema_path': SCHEMA_DIR / 'data_schema.json',
            'best_schema_path': DIR / 'metrics' / f'data_schema_{name}_best_fts.json',
            'feature_importance_path': DIR / 'metrics' / f'feature_importance_{name}.csv',
            'final_feature_importance_path': DIR / 'metrics' / f'feature_importance_{name}_final_features.csv',
            'dropped_features_path': DIR / 'metrics' / f'drop_features_{name}.csv',
            'cv_forward_paths': {
                label: DIR / 'img' / f'cv_forward_{label}_{name}_feature_selector.png'
                for label in self.CV_METRIC_LABELS
            },
        }
        self.__dict__.update(paths)
