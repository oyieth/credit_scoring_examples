import numpy as np
import matplotlib as plt
import seaborn as sns

def correlations(df, do_not_consider, ax = None):
    """
    Triangular correlation plot for df.

    :param df: dataframe with data
    :param do_not_consider: columns not to consider
    :return:
    """
    types = {col: str(df[col].dtype) for col in df.columns}
    columns = [item for item in types if types[item] not in ('object', 'str') and item not in do_not_consider]
    mask = np.triu(np.ones_like(df[columns].corr()))

    # plotting a triangle correlation heatmap
    if ax is None:
        fig, ax = plt.subplots(1, 1, )
        fig.set_size_inches(10, 10)
        fig.set_size_inches(10, 10)
    dataplot = sns.heatmap(df[columns].corr().round(2), cmap="coolwarm", annot=False, mask=mask, ax=ax, vmax=1, vmin=-1)