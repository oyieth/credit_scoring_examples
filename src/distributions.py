import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def group_data_dist(df : pd.DataFrame,
                 col : str,
                 target : str = None) -> pd.DataFrame:

    """
    Distribution of col in dataframe,
    """
    if target is None:
        group = df.groupby([col])[col].count()
        group = pd.DataFrame(group).rename(columns = {col:"N"}).reset_index()
        group["PCT"] = group["N"]/ df[col].count()
    else:
        group = df.groupby([col])[target].sum()
        group_total = df.groupby([col])[target].count().reset_index().rename(columns = {target:"total"})

        group = pd.DataFrame(group).rename(columns = {target:"N"}).reset_index()
        group["PCT_target"] = group["N"] / group_total["total"]
        group["density"] = group["N"] / df[col].count()
    return group

def pie_distribution(df,col):
    """
    Distribution of col in dataframe,
    :param df: pandas dataframe
    :param col: str column name
    :return: plot with pie of  distribution and format
    """
    g = group_data_dist(
            df = df,
            col = col,
            ).set_index(col).plot.pie(x="TARGET",
                       y="PCT", autopct='%1.0f%%',)
    g.get_legend().remove()
    g.axis('off')
    g.set_title(f"{col} distribution");
    return g

def target_barchart(df, chart_col, target_col, ax = None):
    """
    Distribution of target column by column in dataframe,
    :param df: pandas dataframe
    :param chart_col: str column name
    :param target_col: str target column
    :return: bar plot with bar chart of target distribution

    """
    if ax is None:
        fig, ax = plt.subplots()
    dist = group_data_dist(
            df = df,
            col = chart_col,
            target = target_col,

            )
    g = sns.barplot(data = dist,
            x = chart_col,
            y = "PCT_target",
            hue = "density",
            ax = ax)
    g.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    handles, labels = g.get_legend_handles_labels()
    new_labels = [f"{float(label):.2f}%" for label in labels]
    g.legend(handles, new_labels, title="Percentage")
    g.set_title(f"{target_col} density in {target_col}")
    g.tick_params(axis='x', labelrotation=90)
    return g