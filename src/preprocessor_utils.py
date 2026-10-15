import numpy as np
import pandas as pd

import json
import joblib

#Pipeline elements
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.outliers import ArbitraryOutlierCapper
from sklearn.impute import SimpleImputer


from sklearn.preprocessing import FunctionTransformer
from functools import partial

from src.model_paths import ReadWritePaths
from src.transform import ManualSelector, TypeConverter
from src.utils import  get_feature_lists,recast_numeric, load_data_schema


def create_initial_pipeline(data_dict,
                            max_cap = True,
                            min_cap = True):
    """
     Create an initial preprocessing pipeline based on the data schema.

     Args:
         data_dict (dict): A dictionary specifying feature types and imputation rules,
                           where keys are feature names and values are lists of attributes.
         scaler (str): Name of the scaler to be applied to numerical features.

     Returns:
         tuple: A tuple containing:
             - Pipeline: A scikit-learn pipeline object with the defined preprocessing steps.
             - numerical_features (list): List of numerical feature names for further processing.
    """
    (
         features,
         feature_names
    ) = get_feature_lists(data_dict)
    steps = [
         ('manual_selector', ManualSelector(variables=feature_names)),
         ('type_converter', TypeConverter(
             numerical_features = features["numerical"],
             categorical_features = features["categorical"],

         )),
         ('categorical_imputer', CategoricalImputer(
             variables=features["categorical"],
             imputation_method='missing',
             fill_value='ND',
             ignore_format=True,
             return_object=True
         ))]
    #Impute values
    for value in features["impute"]:
        steps = steps + [
            (f'simple_imputer_{value}', SklearnTransformerWrapper(
             SimpleImputer(
                            missing_values=np.nan,
                            strategy='constant',
                            fill_value= value,
                            keep_empty_features=True),
             variables=features["impute"][value]
         ))
        ]
    steps = steps + [
     ('numeric_recast', FunctionTransformer(
    partial(recast_numeric, columns=features["numerical"])
        ))]
    # Optional caps (either, both or none)
    if max_cap or min_cap:
        steps = steps + [('outlier_capper',
                          ArbitraryOutlierCapper(
                              max_capping_dict=features["max"] if max_cap else None,
                              min_capping_dict=features["min"] if min_cap else None,
                          ))]

    steps = steps + [('rare_label_encoder', RareLabelEncoder(
             variables= features["categorical"], tol=0.012, n_categories=5,
             max_n_categories=20, replace_with='OTHER', ignore_format=True
         )),
         ('ordinal_encoder', OrdinalEncoder(
             variables=features["categorical"], encoding_method='ordered',
             ignore_format=True
         ))
                    ]
    steps = steps + [('simple_imputer', SklearnTransformerWrapper(
             SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1),
             variables=feature_names
    ))]

    return Pipeline(steps, verbose=True), features["numerical"]
def get_preprocessor(
        x: pd.DataFrame,
        y: pd.Series,
        target_col: str,
        other_metrics : list,
        data_schema_path: str,
        scaler='StandardScaler',
        write_paths: object = None
):
    """
    Create and save a preprocessing pipeline for feature engineering and selection.

    This function constructs a preprocessing pipeline that includes data type conversion,
    feature imputation, scaling, and optional feature selection based on importance scores.
    It processes the input features (X) using the specified steps and saves the pipeline
    and encoding dictionaries to the provided paths.

    Args:
        x (pd.DataFrame): Input feature matrix.
        y (pd.Series): Target variable.
        data_schema_path (str): Path to the data schema JSON file.

        n_estimators (int): Number of estimators for ensemble methods. Defaults to 100.
        max_features (int): Maximum number of features to select.
        threshold (int): Threshold for feature importance scores. Defaults to 0.
        scaler (str): Scaling method to apply to numerical features .
        random_state (int): Random seed for reproducibility. Defaults to 1112.
        write_paths (object): An object containing paths for saving the pipeline.

    Returns:
        final_pipeline (Pipeline): A scikit-learn pipeline object with preprocessing
            and feature selection steps.
        column_names (list): List of feature names selected or retained after preprocessing.
    """
    data_dict = load_data_schema(data_schema_path)
    scaler_ = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "None": None
    }[scaler]

    initial_pipeline, numerical_features = create_initial_pipeline(
        data_dict
    )
    feature_names = list(data_dict.keys())
    initial_pipeline.fit(x, y)
    ordinal_encoder_dict = getattr(initial_pipeline['ordinal_encoder'], 'encoder_dict_', {})

    x_preprocessed = initial_pipeline.transform(x)
    print('x_preprocessed inital pipeline: Done')

    if scaler_:
        scaler_pipeline = Pipeline(
            [('scaler', SklearnTransformerWrapper(scaler_, variables=numerical_features))],
            verbose=True
        )
        scaler_pipeline.fit(x_preprocessed)
        x_preprocessed = scaler_pipeline.transform(x_preprocessed)

        initial_pipeline = Pipeline([
            ('initial_pipeline', initial_pipeline),
            ('scaler', scaler_pipeline)
        ], verbose=True)

    final_pipeline = initial_pipeline
    column_names = feature_names
    loc_dir = data_schema_path.parent
    schema_file = data_schema_path.name

    new_schema_file = f'{schema_file.split(".")[0]}_train.{schema_file.split(".")[1]}'
    assert target_col not in column_names
    for item in other_metrics:
        assert item not in column_names
    if isinstance(write_paths, ReadWritePaths):
        with open(loc_dir / new_schema_file,
                  'w',
                  encoding='utf-8') as file_schema:
            json.dump(data_dict, file_schema, separators=(',', ':'), indent=2)
        joblib.dump(final_pipeline,
                    write_paths.preprocessor_path)

        with open(write_paths.ordinal_encoder_dict_path, 'w', encoding='utf-8') as file_encoder:
            json.dump(ordinal_encoder_dict, file_encoder, indent=4)

        print(
            f'Preprocessor saved at {write_paths.preprocessor_path} '
            f'& Ordinal encoder dict saved at {write_paths.ordinal_encoder_dict_path}!'
        )
    else:
        print("No write_paths provided")
    return final_pipeline, column_names


def save_preprocessed_data(
        x, y, other_metrics, preprocessor, column_names, target, path, save=True):
    """
    Save the preprocessed dataset.

    This function applies the provided preprocessor to the input features
    (X) and combines the transformed data with the target variable (y)
    and any specified other metrics. The resulting DataFrame is then
    saved as a CSV file at the specified path if `save` is set to True.
    If `save` is False, the function returns the preprocessed DataFrame
    instead of saving it.

    Args:
        x (pd.DataFrame): The input features to be preprocessed.
        y (pd.Series): The target variable associated with the input features.
        other_metrics (list): A list of other metrics to include in the final DataFrame.
        preprocessor: The preprocessor object used to transform the input features.
        column_names (list): The names of the columns in the transformed DataFrame.
        target (str): The name of the target variable to include in the final DataFrame.
        path (str): The file path where the preprocessed dataset will be saved.
        save (bool): Flag indicating whether to save the DataFrame as a CSV file.

    Returns:
        pd.DataFrame or None: Returns the preprocessed DataFrame if `save` is False;
                              otherwise, returns None.
    """
    data_preprocessed = pd.DataFrame(
        preprocessor.transform(x), columns=column_names, index=x.index)
    data_preprocessed[target] = y
    if other_metrics is None:
        other_metrics = []
    data_preprocessed[other_metrics] = x[other_metrics]
    data_preprocessed = data_preprocessed[other_metrics + [target] + column_names]
    if save:
        data_preprocessed.to_csv(path, index=False)
        print(f'Dataset saved at {path}!')

    return data_preprocessed


def get_column_names(preprocessor):
    """
    Retrieve column names from a preprocessing object.

    Parameters:
    ----------
    preprocessor : object
        The preprocessor object, which is expected to contain methods or attributes
        that provide the names of the processed features.

    Returns:
    -------
    list
        A list of column names derived from the preprocessor.
    """
    try:
        return preprocessor[-1].get_feature_names_out()
    except AttributeError:
        pass

    for attr in ['out_var', 'variables', 'poly_features', 'best_features']:
        try:
            return getattr(preprocessor[0], attr, preprocessor[0].variables)
        except AttributeError:
            continue

    raise RuntimeError("Unable to determine column names from preprocessor.")