import altair
import pandas as pd
import streamlit as st
import sys
import altair as alt
import os


PROJECT_DIR = "/Users/oyieth/Documents/Python projects/credit_scoring_examples"
sys.path.append(str(f"{PROJECT_DIR}/"))
from src.distributions import group_data_dist
#src/distributions.py
datasets = {
    "train": f"{PROJECT_DIR}/data/application_train.csv",
    "test": f"{PROJECT_DIR}/data/application_test.csv",
    "bureau": f"{PROJECT_DIR}/data/bureau.csv",
    "bureau_balance": f"{PROJECT_DIR}/data/bureau_balance.csv",
    "credit_card_balance": f"{PROJECT_DIR}/data/credit_card_balance.csv",
    "installments_payments": f"{PROJECT_DIR}/data/installments_payments.csv",
    "POS_CASH_balance": f"{PROJECT_DIR}/data/POS_CASH_balance.csv",
    "previous_application": f"{PROJECT_DIR}/data/previous_application.csv",
}
descriptions_path = f"{PROJECT_DIR}/data/HomeCredit_columns_description.csv"
descriptions = pd.read_csv(descriptions_path, encoding = 'latin')[["Row", "Description"]]
descriptions = dict(zip(descriptions["Row"].values, descriptions["Description"].values))
""" # Data exploration"""
selection = st.radio(
    "Dataset",
    datasets.keys(),
)
if selection:
    df = pd.read_csv(datasets[selection])

    # Target to model
    target = "TARGET"
    # ID columns in the datasat
    id_cols = ["SK_ID_CURR",
           "SK_BUREAU_ID",
           "SK_ID_PREV"
               "SK_ID_BUREAU",
           ]
    do_not_consider = [target] + id_cols


    types = {col:str(df[col].dtype) for  col in df.columns}

    cat_tab, num_tab = st.tabs(["Categorical", "Numerical"])
    with cat_tab:
        st.header("Categorical features.")
        options = [item for item in types if types[item] == 'str' and item not in do_not_consider]
        if len(options) > 0:
            feature_cat = st.menu_button("Feature",
                         options,)
            if feature_cat:
                if feature_cat in descriptions:
                    description_feature = descriptions[feature_cat]
                else:
                    description_feature = f"{feature_cat} not in descriptions"
                f"""{description_feature}"""
                group = group_data_dist(df, feature_cat)
                base = alt.Chart(group.sort_values(by = "PCT")).encode(
                    theta=alt.Theta("PCT", title="%", type="quantitative"),
                    color=altair.Color(feature_cat).scale(scheme='set1'),
                    tooltip=[
                        alt.Tooltip(feature_cat),
                        alt.Tooltip("N", title="N", format=",d"),
                        alt.Tooltip("PCT", title="%", format=".0%"),
                         ]
                        )
                pie_chart = base.mark_arc(innerRadius=50, outerRadius=120).encode()
                text = base.mark_text(radius=140, size=20).encode(altair.Text("PCT", format=".0%"))
                st.altair_chart(pie_chart + text )

                if target in df.columns:
                    target_dist = group_data_dist(df, feature_cat, target = target)
                    bar_chart = alt.Chart(target_dist.sort_values(by = "PCT_target",ascending= False)).mark_bar().encode(
                        x = feature_cat,
                        y = alt.Y("PCT_target",
                                  axis=alt.Axis(format="%"),
                                  ),
                        color = 'density',
                        tooltip=[
                            alt.Tooltip(feature_cat),
                            alt.Tooltip("PCT_target", title="% with target", format=".0%"),
                            alt.Tooltip("density", title="% target from whole population", format=".0%"),
                                ]
                    )

                    st.altair_chart(bar_chart)
    with num_tab:
        st.header("Numerical features.")
        options = [item for item in types if types[item] != 'str' and item not in do_not_consider]
        if len(options) >0:
            feature_num = st.menu_button("Feature", options)
            if feature_num:
                if feature_num in descriptions:
                    description_feature = descriptions[feature_num]
                else:
                    description_feature = f"{feature_num} not in descriptions"
                f"""{description_feature}"""

                hist = alt.Chart(df[[feature_num]]).mark_bar().encode(
                alt.X(feature_num,
                      bin=alt.BinParams(maxbins = 50)),
                    y='count()',
                )
                col1, col2 =st.columns([1,3])
                col1.dataframe(df[[feature_num]].describe(percentiles=[0.05, .25, .5, .75, .95, .99]),
                           column_config= {feature_num: st.column_config.NumberColumn(format = "%,.2d")},
                           )
                col2.altair_chart(hist)
