
import pandas as pd
from src.utils import convert_numerical_features, convert_categorical_features
class Converter:
    """
    Original class for transformers and converters.
    """

    def __init__(self, variables: list[str]):
        self

    def transform(self, X) -> pd.DataFrame:
        """
        Transform method for compatibility with scikit-learn pipelines.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        --------
        pd.DataFrame
            This instance does nothing for the dataframe.
        """
        return X

    def fit(self, X: pd.DataFrame, y=None) -> 'ManualSelector':
        """
        Fit method for compatibility with scikit-learn pipelines.
        """
        return self


class ManualSelector(Converter):
    """
    A transformer for manually selecting specified variables from a DataFrame.

    Parameters:
    -----------
    variables : List[str]
        The list of variables to select.

    Methods:
    --------
    transform(X: pd.DataFrame) -> pd.DataFrame
        Selects the specified variables from the DataFrame.

    fit(X: pd.DataFrame, y=None) -> ManualSelector
        Fits the transformer, returns self.
    """

    def __init__(self, variables: list[str]):
        self.variables = variables

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select specified variables from the input DataFrame.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing only the selected variables.
        """
        return X[self.variables]


class TypeConverter(Converter):
    """
    A transformer to convert data types of numerical, boolean, and categorical features
    in a DataFrame, including imputations for missing values.

    Parameters:
    -----------
    numerical_features : list[str]
        list of numerical feature names.
    categorical_features : list[str]
        List of categorical feature names.

    Methods:
    --------
    transform(X: pd.DataFrame) -> pd.DataFrame
        Transforms the data types of the input DataFrame.
    fit(X: pd.DataFrame, y=None) -> TypeConverter
        Fits the transformer, returns self.
    """

    def __init__(
            self,
            numerical_features: list[str],
            categorical_features: list[str],
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data types of numerical, boolean, and categorical features.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with updated data types.
        """
        # Converting numerical features to float64
        X = convert_numerical_features(X, self.numerical_features)

        # Converting categorical variables to object and cleaning categories
        X = convert_categorical_features(X, self.categorical_features)

        return X
