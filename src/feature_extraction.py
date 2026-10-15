import numpy as np
import pandas as pd

def mode_column(df : pd.DataFrame,
                col : str,
                id_col : str,
                cols_count : list,
                cols_sum : list):

    """
        Value more repeated by column in the dataframe. This frequency is calculated by
        other aggregated columns.

        :param df: pd.DataFrame with data
        :param col: column of entered
        :param id_col: id column to aggregate
        :param cols_count: columns to agreggate by count
        :param cols_sum: columns to aggregate by sum
        :return: most frequent value by aggregation
        """
    agg_method = {}
    for item in cols_count:
        agg_method[item] = 'count'
    for item in cols_sum:
        agg_method[item] = 'sum'
    modes_type = df.groupby(by=[id_col, col], as_index=False).agg(agg_method)
    modes_type = modes_type.sort_values([id_col] + list(agg_method),
                                            ascending=[True] + [False for item in list(agg_method)])
    modes_type['row_num'] = modes_type.groupby(id_col).cumcount() + 1
    modes_type = modes_type[modes_type["row_num"] == 1].set_index(id_col)[col]
    return modes_type

def summarize_installments(id_prev_applications : pd.DataFrame,
                           installments_payments : pd.DataFrame):
    """
    Summarize installments payments.
    :param id_prev_applications: pd.DataFrame with previous applications ID's:
    :param installments_payments: pd.DataFrame with installments payments
    :return:
    pd.DataFrame with summy
    """
    # Useful installment data
    installments_payments["pct_payment"] = (
    installments_payments["AMT_PAYMENT"] / installments_payments["AMT_INSTALMENT"]).replace(np.inf, 0)
    installments_payments.loc[(installments_payments["pct_payment"] < 1) & (
    installments_payments["AMT_INSTALMENT"] > 0), "flag_unfullfilled"] = 1
    installments_payments["flag_unfullfilled"] = installments_payments["flag_unfullfilled"].fillna(0)
    installments_payments.loc[(installments_payments["pct_payment"] == 0) & (
    installments_payments["AMT_INSTALMENT"] > 0), "flag_unpayment"] = 1
    installments_payments["flag_unpayment"] = installments_payments["flag_unpayment"].fillna(0)

    summary_installment = id_prev_applications.set_index("SK_ID_PREV")
    summary_installment["n_versions"] = installments_payments.groupby(by="SK_ID_PREV").NUM_INSTALMENT_VERSION.nunique()
    summary_installment["total_installments"] = installments_payments.groupby(
                                                by="SK_ID_PREV").NUM_INSTALMENT_NUMBER.max()
    summary_installment["latest_day"] = installments_payments.groupby(by="SK_ID_PREV").DAYS_INSTALMENT.max()
    summary_installment["first_day"] = installments_payments.groupby(by="SK_ID_PREV").DAYS_INSTALMENT.min()

    summary_installment["latest_payment"] = installments_payments.groupby(by="SK_ID_PREV").DAYS_ENTRY_PAYMENT.max()
    summary_installment["first_payment"] = installments_payments.groupby(by="SK_ID_PREV").DAYS_ENTRY_PAYMENT.min()

    summary_installment["max_installment"] = installments_payments.groupby(by="SK_ID_PREV").AMT_INSTALMENT.max()
    summary_installment["min_installment"] = installments_payments.groupby(by="SK_ID_PREV").AMT_INSTALMENT.min()
    summary_installment["mean_installment"] = installments_payments.groupby(by="SK_ID_PREV").AMT_INSTALMENT.mean()
    summary_installment["mean_pay_pct_installment"] = installments_payments.groupby(by="SK_ID_PREV").pct_payment.mean()
    summary_installment["min_pay_pct_installment"] = installments_payments.groupby(by="SK_ID_PREV").pct_payment.min()
    summary_installment["max_pay_pct_installment"] = installments_payments.groupby(by="SK_ID_PREV").pct_payment.max()
    summary_installment["times_unfullfilled"] = installments_payments.groupby(by="SK_ID_PREV").flag_unfullfilled.sum()
    summary_installment["times_unpayed"] = installments_payments.groupby(by="SK_ID_PREV").flag_unpayment.sum()
    return summary_installment


def summarize_cash_balance(id_prev_applications,
                          pos_cash_balance ):
    pos_summary = id_prev_applications.set_index("SK_ID_PREV")
    pos_summary["n_months"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["MONTHS_BALANCE"].count()

    pos_summary["total_prev_terms"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["CNT_INSTALMENT"].nunique()
    pos_summary["min_prev_term"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["CNT_INSTALMENT"].min()
    pos_summary["max_prev_term"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["CNT_INSTALMENT"].max()
    pos_summary["max_remaining"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["CNT_INSTALMENT_FUTURE"].max()
    pos_summary["number_status"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["NAME_CONTRACT_STATUS"].nunique()
    pos_summary["total_dpd"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["SK_DPD"].sum()
    pos_summary["total_dpd_def"] = pos_cash_balance.groupby(by = "SK_ID_PREV")["SK_DPD_DEF"].sum()
    pos_summary["flag_dpd"] = pos_summary["total_dpd"].apply(lambda x: 1 if x>0 else 0)
    pos_summary["flag_dpd_def"] = pos_summary["total_dpd"].apply(lambda x: 1 if x>0 else 0)
    latest_month = pos_cash_balance.groupby("SK_ID_PREV", as_index = False)["MONTHS_BALANCE"].max()
    pos_summary["latest_status"] = latest_month.merge(pos_cash_balance[["SK_ID_PREV","MONTHS_BALANCE","NAME_CONTRACT_STATUS"]],
                   on = ["SK_ID_PREV","MONTHS_BALANCE"])[["SK_ID_PREV", "NAME_CONTRACT_STATUS"]
                ].rename(columns = {"NAME_CONTRACT_STATUS":"last_status"}).set_index('SK_ID_PREV')
    return pos_summary

def summarize_credit_card(id_prev_applications,
                          credit_card):
    cc_summary = id_prev_applications.set_index("SK_ID_PREV")

    cc_summary["n_months_cc"] = credit_card.groupby(by="SK_ID_PREV")["MONTHS_BALANCE"].count()

    cc_summary["mean_amt_balance"] = credit_card.groupby(by="SK_ID_PREV")["AMT_BALANCE"].mean()
    cc_summary["max_amt_balance"] = credit_card.groupby(by="SK_ID_PREV")["AMT_BALANCE"].max()
    cc_summary["min_amt_balance"] = credit_card.groupby(by="SK_ID_PREV")["AMT_BALANCE"].min()

    cc_summary["mean_amt_limit"] = credit_card.groupby(by="SK_ID_PREV")["AMT_CREDIT_LIMIT_ACTUAL"].mean()
    cc_summary["max_amt_limit"] = credit_card.groupby(by="SK_ID_PREV")["AMT_CREDIT_LIMIT_ACTUAL"].max()
    cc_summary["min_amt_limit"] = credit_card.groupby(by="SK_ID_PREV")["AMT_CREDIT_LIMIT_ACTUAL"].min()

    cc_summary["mean_amt_drawing_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_ATM_CURRENT"].mean()
    cc_summary["max_amt_drawing_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_ATM_CURRENT"].max()
    cc_summary["min_amt_drawing_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_ATM_CURRENT"].min()

    cc_summary["mean_amt_drawing_other_current"] = credit_card.groupby(by="SK_ID_PREV")[
        "AMT_DRAWINGS_OTHER_CURRENT"].mean()
    cc_summary["max_amt_drawing_other_current"] = credit_card.groupby(by="SK_ID_PREV")[
        "AMT_DRAWINGS_OTHER_CURRENT"].max()
    cc_summary["min_amt_drawing_other_current"] = credit_card.groupby(by="SK_ID_PREV")[
        "AMT_DRAWINGS_OTHER_CURRENT"].min()

    cc_summary["mean_amt_drawing_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_POS_CURRENT"].mean()
    cc_summary["max_amt_drawing_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_POS_CURRENT"].max()
    cc_summary["min_amt_drawing_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["AMT_DRAWINGS_POS_CURRENT"].min()

    cc_summary["mean_min_installment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_INST_MIN_REGULARITY"].mean()
    cc_summary["max_min_installment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_INST_MIN_REGULARITY"].max()
    cc_summary["min_min_installment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_INST_MIN_REGULARITY"].min()

    cc_summary["mean_payment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_CURRENT"].mean()
    cc_summary["max_payment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_CURRENT"].max()
    cc_summary["min_payment"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_CURRENT"].min()

    cc_summary["mean_payment_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_TOTAL_CURRENT"].mean()
    cc_summary["max_payment_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_TOTAL_CURRENT"].max()
    cc_summary["min_payment_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_PAYMENT_TOTAL_CURRENT"].min()

    cc_summary["mean_receivable_principal"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECEIVABLE_PRINCIPAL"].mean()
    cc_summary["max_receivable_principal"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECEIVABLE_PRINCIPAL"].max()
    cc_summary["min_receivable_principal"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECEIVABLE_PRINCIPAL"].min()

    cc_summary["mean_receivable"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECIVABLE"].mean()
    cc_summary["max_receivable"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECIVABLE"].max()
    cc_summary["min_receivable"] = credit_card.groupby(by="SK_ID_PREV")["AMT_RECIVABLE"].min()

    cc_summary["mean_receivable_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_TOTAL_RECEIVABLE"].mean()
    cc_summary["max_receivable_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_TOTAL_RECEIVABLE"].max()
    cc_summary["min_receivable_total"] = credit_card.groupby(by="SK_ID_PREV")["AMT_TOTAL_RECEIVABLE"].min()

    cc_summary["mean_drawings_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_CURRENT"].mean()
    cc_summary["max_drawings_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_CURRENT"].max()
    cc_summary["min_drawings_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_CURRENT"].min()

    cc_summary["mean_drawings_other_current"] = credit_card.groupby(by="SK_ID_PREV")[
        "CNT_DRAWINGS_OTHER_CURRENT"].mean()
    cc_summary["max_drawings_other_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_OTHER_CURRENT"].max()
    cc_summary["min_drawings_other_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_OTHER_CURRENT"].min()

    cc_summary["mean_drawings_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_POS_CURRENT"].mean()
    cc_summary["max_drawings_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_POS_CURRENT"].max()
    cc_summary["min_drawings_pos_current"] = credit_card.groupby(by="SK_ID_PREV")["CNT_DRAWINGS_POS_CURRENT"].min()

    cc_summary["mean_installments_mature"] = credit_card.groupby(by="SK_ID_PREV")["CNT_INSTALMENT_MATURE_CUM"].mean()
    cc_summary["max_installments_mature"] = credit_card.groupby(by="SK_ID_PREV")["CNT_INSTALMENT_MATURE_CUM"].max()
    cc_summary["min_installments_mature"] = credit_card.groupby(by="SK_ID_PREV")["CNT_INSTALMENT_MATURE_CUM"].min()

    cc_summary["total_dpd"] = credit_card.groupby(by="SK_ID_PREV")["SK_DPD"].sum()
    cc_summary["total_dpd_def"] = credit_card.groupby(by="SK_ID_PREV")["SK_DPD_DEF"].sum()
    cc_summary["flag_dpd"] = cc_summary["total_dpd"].apply(lambda x: 1 if x > 0 else 0)
    cc_summary["flag_dpd_def"] = cc_summary["total_dpd"].apply(lambda x: 1 if x > 0 else 0)
    latest_month = credit_card.groupby("SK_ID_PREV", as_index=False)["MONTHS_BALANCE"].max()
    cc_summary["latest_status"] = \
    latest_month.merge(credit_card[["SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS"]],
                       on=["SK_ID_PREV", "MONTHS_BALANCE"])[["SK_ID_PREV", "NAME_CONTRACT_STATUS"]
    ].rename(columns={"NAME_CONTRACT_STATUS": "last_status"}).set_index('SK_ID_PREV')
    return cc_summary


def get_previous_application_features(id_loans,
                                      prev_application,
                                      installments_payments,
                                      pos_cash_balance,
                                      credit_card,):
    features_previous = id_loans[['SK_ID_CURR']].set_index('SK_ID_CURR')
    features_previous_instlmnt = features_previous.copy()
    features_previous_cash = features_previous.copy()
    features_previous_cc = features_previous.copy()
    # Cleanup prev application
    prev_application["flag_canceled"] = prev_application["NAME_CONTRACT_STATUS"].apply(
        lambda x: 1 if x == 'Canceled' else 0)
    prev_application["flag_refused"] = prev_application["NAME_CONTRACT_STATUS"].apply(
        lambda x: 1 if x == 'Refused' else 0)
    prev_application["flag_unused"] = prev_application["NAME_CONTRACT_STATUS"].apply(
        lambda x: 1 if x == 'Unused offer' else 0)

    prev_application["flag_down_payment"] = prev_application["RATE_DOWN_PAYMENT"].apply(
        lambda x: 0 if pd.isna(x) else 1)
    prev_application["flag_interest"] = prev_application["RATE_INTEREST_PRIMARY"].apply(
        lambda x: 0 if pd.isna(x) else 1)
    for item in ['NAME_CASH_LOAN_PURPOSE',
                 'NAME_PAYMENT_TYPE',
                 'NAME_GOODS_CATEGORY',
                 'NAME_PORTFOLIO',
                 'NAME_PRODUCT_TYPE',
                 'NAME_SELLER_INDUSTRY',
                 'NAME_YIELD_GROUP',
                 'PRODUCT_COMBINATION',
                 ]:
        prev_application[item] = prev_application[item].replace("XNA", np.nan)
    prev_application["FLAG_LAST_APPL_PER_CONTRACT"] = prev_application["FLAG_LAST_APPL_PER_CONTRACT"].apply(
        lambda x: 1 if x == 'Y' else 0)

    # Numeric columns
    # number of previous aplication
    features_previous["previous_applications"] = prev_application.groupby(by="SK_ID_CURR")["SK_ID_PREV"].nunique()
    # Diferent types in previous applicacion
    features_previous["n_types_applications"] = prev_application.groupby(by="SK_ID_CURR")[
        "NAME_CONTRACT_TYPE"].nunique()
    # Different weekday start of application
    features_previous["n_weekdays"] = prev_application.groupby(by="SK_ID_CURR")['WEEKDAY_APPR_PROCESS_START'].nunique()
    # Different hours when starting previous application
    features_previous["n_hour_start"] = prev_application.groupby(by="SK_ID_CURR")['HOUR_APPR_PROCESS_START'].nunique()
    # Number of different purposes of application
    features_previous["n_purpose"] = prev_application.groupby(by="SK_ID_CURR")['NAME_CASH_LOAN_PURPOSE'].nunique()
    # Number of different status for applicacion
    features_previous["n_status"] = prev_application.groupby(by="SK_ID_CURR")['NAME_CONTRACT_STATUS'].nunique()
    # Number of payment types for previous application
    features_previous["n_payment_type"] = prev_application.groupby(by="SK_ID_CURR")['NAME_PAYMENT_TYPE'].nunique()
    # Number of code rejections for previous application
    features_previous["n_code_rejection"] = prev_application.groupby(by="SK_ID_CURR")['CODE_REJECT_REASON'].nunique()
    # Number of type suites for previous application
    features_previous["n_suite"] = prev_application.groupby(by="SK_ID_CURR")['NAME_TYPE_SUITE'].nunique()
    # Number of client types for previous application
    features_previous["n_client_type"] = prev_application.groupby(by="SK_ID_CURR")['NAME_CLIENT_TYPE'].nunique()
    # Number of names for goods category for previous application
    features_previous["n_goods_category"] = prev_application.groupby(by="SK_ID_CURR")['NAME_GOODS_CATEGORY'].nunique()
    # Number of portfolios for previous application
    features_previous["n_portfolio"] = prev_application.groupby(by="SK_ID_CURR")['NAME_PORTFOLIO'].nunique()
    # Number of product types for previous application
    features_previous["n_product_type"] = prev_application.groupby(by="SK_ID_CURR")['NAME_PRODUCT_TYPE'].nunique()
    # Number of channel types for previous application
    features_previous["n_channel_type"] = prev_application.groupby(by="SK_ID_CURR")['CHANNEL_TYPE'].nunique()
    # Number of seller areas for previous application
    features_previous["n_sellerplace_area"] = prev_application.groupby(by="SK_ID_CURR")['SELLERPLACE_AREA'].nunique()
    # Number of seller industries for previous application
    features_previous["n_seller_industry"] = prev_application.groupby(by="SK_ID_CURR")['NAME_SELLER_INDUSTRY'].nunique()
    # Number for yield groups for previous application
    features_previous["n_yield_groups"] = prev_application.groupby(by="SK_ID_CURR")['NAME_YIELD_GROUP'].nunique()
    # Number of product combinations
    features_previous["n_product_comb"] = prev_application.groupby(by="SK_ID_CURR")['PRODUCT_COMBINATION'].nunique()

    # Amounts: Continuous distributions.mean, mix, max and median.
    features_previous["amt_annuity_mean"] = prev_application.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].mean()
    features_previous["amt_annuity_min"] = prev_application.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].min()
    features_previous["amt_annuity_max"] = prev_application.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].max()
    features_previous["amt_annuity_median"] = prev_application.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].median()

    features_previous["amt_application_mean"] = prev_application.groupby(by="SK_ID_CURR")["AMT_APPLICATION"].mean()
    features_previous["amt_application_min"] = prev_application.groupby(by="SK_ID_CURR")["AMT_APPLICATION"].min()
    features_previous["amt_application_max"] = prev_application.groupby(by="SK_ID_CURR")["AMT_APPLICATION"].max()
    features_previous["amt_application_median"] = prev_application.groupby(by="SK_ID_CURR")["AMT_APPLICATION"].median()

    features_previous["amt_credit_mean"] = prev_application.groupby(by="SK_ID_CURR")["AMT_CREDIT"].mean()
    features_previous["amt_credit_min"] = prev_application.groupby(by="SK_ID_CURR")["AMT_CREDIT"].min()
    features_previous["amt_credit_max"] = prev_application.groupby(by="SK_ID_CURR")["AMT_CREDIT"].max()
    features_previous["amt_credit_median"] = prev_application.groupby(by="SK_ID_CURR")["AMT_CREDIT"].median()

    features_previous["amt_down_payment_mean"] = prev_application.groupby(by="SK_ID_CURR")["AMT_DOWN_PAYMENT"].mean()
    features_previous["amt_down_payment_min"] = prev_application.groupby(by="SK_ID_CURR")["AMT_DOWN_PAYMENT"].min()
    features_previous["amt_down_payment_max"] = prev_application.groupby(by="SK_ID_CURR")["AMT_DOWN_PAYMENT"].max()
    features_previous["amt_down_payment_median"] = prev_application.groupby(by="SK_ID_CURR")[
        "AMT_DOWN_PAYMENT"].median()

    features_previous["amt_goods_mean"] = prev_application.groupby(by="SK_ID_CURR")["AMT_GOODS_PRICE"].mean()
    features_previous["amt_goods_min"] = prev_application.groupby(by="SK_ID_CURR")["AMT_GOODS_PRICE"].min()
    features_previous["amt_goods_max"] = prev_application.groupby(by="SK_ID_CURR")["AMT_GOODS_PRICE"].max()
    features_previous["amt_goods_median"] = prev_application.groupby(by="SK_ID_CURR")["AMT_GOODS_PRICE"].median()

    features_previous["cnt_payment_mean"] = prev_application.groupby(by="SK_ID_CURR")["CNT_PAYMENT"].mean()
    features_previous["cnt_payment_min"] = prev_application.groupby(by="SK_ID_CURR")["CNT_PAYMENT"].min()
    features_previous["cnt_payment_max"] = prev_application.groupby(by="SK_ID_CURR")["CNT_PAYMENT"].max()
    features_previous["cnt_payment_median"] = prev_application.groupby(by="SK_ID_CURR")["CNT_PAYMENT"].median()

    features_previous["down_payment_mean"] = prev_application.groupby(by="SK_ID_CURR")["RATE_DOWN_PAYMENT"].mean()
    features_previous["down_payment_min"] = prev_application.groupby(by="SK_ID_CURR")["RATE_DOWN_PAYMENT"].min()
    features_previous["down_payment_max"] = prev_application.groupby(by="SK_ID_CURR")["RATE_DOWN_PAYMENT"].max()
    features_previous["down_payment_median"] = prev_application.groupby(by="SK_ID_CURR")["RATE_DOWN_PAYMENT"].median()

    features_previous["rate_interested_primary_mean"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIMARY"].mean()
    features_previous["rate_interested_primary_min"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIMARY"].min()
    features_previous["rate_interested_primary_max"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIMARY"].max()
    features_previous["rate_interested_primary_median"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIMARY"].median()

    features_previous["rate_intereseted_privileged_mean"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIVILEGED"].mean()
    features_previous["rate_intereseted_privileged_min"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIVILEGED"].min()
    features_previous["rate_intereseted_privileged_max"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIVILEGED"].max()
    features_previous["rate_intereseted_privileged_median"] = prev_application.groupby(by="SK_ID_CURR")[
        "RATE_INTEREST_PRIVILEGED"].median()

    # Days: Min and Max only (Oldest and more recent)

    features_previous["days_decision_min"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_DECISION"].min()
    features_previous["days_decision_max"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_DECISION"].max()

    features_previous["days_first_drawing_min"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_FIRST_DRAWING"].min()
    features_previous["days_first_drawing_max"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_FIRST_DRAWING"].max()

    features_previous["days_first_due_min"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_FIRST_DUE"].min()
    features_previous["days_first_due_max"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_FIRST_DUE"].max()

    features_previous["days_last_due_first_min"] = prev_application.groupby(by="SK_ID_CURR")[
        "DAYS_LAST_DUE_1ST_VERSION"].min()
    features_previous["days_last_due_first_max"] = prev_application.groupby(by="SK_ID_CURR")[
        "DAYS_LAST_DUE_1ST_VERSION"].max()

    features_previous["days_last_due_min"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_LAST_DUE"].min()
    features_previous["days_last_due_max"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_LAST_DUE"].max()

    features_previous["days_termination_min"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_TERMINATION"].min()
    features_previous["days_termination_max"] = prev_application.groupby(by="SK_ID_CURR")["DAYS_TERMINATION"].max()

    # Mean
    features_previous["mean_down_payment"] = prev_application.groupby(by="SK_ID_CURR")["RATE_DOWN_PAYMENT"].mean()
    # Sum (Mostly flags)
    features_previous["n_canceled"] = prev_application.groupby(by="SK_ID_CURR")["flag_canceled"].sum()
    features_previous["n_refused"] = prev_application.groupby(by="SK_ID_CURR")["flag_refused"].sum()
    features_previous["n_unused"] = prev_application.groupby(by="SK_ID_CURR")["flag_unused"].sum()
    features_previous["n_down_payment"] = prev_application.groupby(by="SK_ID_CURR")["flag_down_payment"].sum()
    features_previous["n_interest"] = prev_application.groupby(by="SK_ID_CURR")["flag_interest"].sum()

    features_previous["n_last_apply_day"] = prev_application.groupby(by="SK_ID_CURR")["NFLAG_LAST_APPL_IN_DAY"].sum()
    features_previous["n_insured_approval"] = prev_application.groupby(by="SK_ID_CURR")[
        "NFLAG_INSURED_ON_APPROVAL"].sum()
    features_previous["n_last_apply_contract"] = prev_application.groupby(by="SK_ID_CURR")[
        "FLAG_LAST_APPL_PER_CONTRACT"].sum()

    # Categorical columns: most frequent feature used
    columns_cat = [
        'NAME_CONTRACT_TYPE',
        'WEEKDAY_APPR_PROCESS_START',
        'HOUR_APPR_PROCESS_START',
        'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS',
        'NAME_PAYMENT_TYPE',
        'CODE_REJECT_REASON',
        'NAME_TYPE_SUITE',
        'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO',
        'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE',
        'SELLERPLACE_AREA',
        'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP',
        'PRODUCT_COMBINATION',
    ]
    columns_freq = {
        'NAME_CONTRACT_TYPE': "freq_contract_type",
        'WEEKDAY_APPR_PROCESS_START': "freq_weekday",
        'HOUR_APPR_PROCESS_START': "freq_hour",
        'NAME_CASH_LOAN_PURPOSE': "freq_purpose",
        'NAME_CONTRACT_STATUS': "freq_status",
        'NAME_PAYMENT_TYPE': "freq_payment_type",
        'CODE_REJECT_REASON': "freq_reject",
        'NAME_TYPE_SUITE': "fred_suite",
        'NAME_CLIENT_TYPE': "freq_client_type",
        'NAME_GOODS_CATEGORY': "freq_goods",
        'NAME_PORTFOLIO': "freq_portfolio",
        'NAME_PRODUCT_TYPE': "freq_product",
        'CHANNEL_TYPE': "freq_channel",
        'SELLERPLACE_AREA': "freq_area",
        'NAME_SELLER_INDUSTRY': "freq_industry",
        'NAME_YIELD_GROUP': "freq_yield_group",
        'PRODUCT_COMBINATION': "fred_product_comb",
    }

    for col in columns_cat:
        features_previous[columns_freq[col]] = mode_column(df=prev_application,
                                                           col=col,
                                                           id_col='SK_ID_CURR',
                                                           cols_count=['SK_ID_PREV'],
                                                           cols_sum=['AMT_APPLICATION']
                                                           )

    # Payment installments

    prev_app_detail_instlmnt = (prev_application[["SK_ID_CURR", "SK_ID_PREV"]
                                ].drop_duplicates().set_index('SK_ID_CURR')
                                )
    prev_app_detail_instlmnt = features_previous_instlmnt.merge(prev_app_detail_instlmnt,
                                     left_index=True,
                                     right_index=True,)
    prev_app_detail_instlmnt = prev_app_detail_instlmnt.merge(summarize_installments(prev_app_detail_instlmnt[["SK_ID_PREV"]], installments_payments=installments_payments),
                                                              left_on="SK_ID_PREV",
                                                              right_index=True)

    features_previous_instlmnt["n_installments"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR").n_versions.count()
    features_previous_instlmnt["max_installments"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR").n_versions.max()
    features_previous_instlmnt["min_installments"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR").n_versions.min()
    features_previous_instlmnt["latest_day"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")["latest_day"].max()
    features_previous_instlmnt["first_day"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")["first_day"].min()
    features_previous_instlmnt["latest_payment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "latest_payment"].max()
    features_previous_instlmnt["first_payment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "first_payment"].min()
    features_previous_instlmnt["max_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "max_installment"].max()
    features_previous_instlmnt["min_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "min_installment"].min()
    features_previous_instlmnt["mean_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "mean_installment"].mean()
    features_previous_instlmnt["mean_pay_pct_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "mean_pay_pct_installment"].mean()
    features_previous_instlmnt["min_pay_pct_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "min_pay_pct_installment"].min()
    features_previous_instlmnt["max_pay_pct_installment"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "max_pay_pct_installment"].max()
    features_previous_instlmnt["times_unfullfilled"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "times_unfullfilled"].sum()
    features_previous_instlmnt["times_unpayed"] = prev_app_detail_instlmnt.groupby(by="SK_ID_CURR")[
        "times_unpayed"].sum()

    # POS Cash

    prev_app_detail_cash = (prev_application[["SK_ID_CURR", "SK_ID_PREV"]
                            ].drop_duplicates().set_index('SK_ID_CURR')
                            )
    prev_app_detail_cash = features_previous_cash.merge(prev_app_detail_cash,
                                                                left_index=True,
                                                                right_index=True, )
    prev_app_detail_cash = prev_app_detail_cash.merge(
        summarize_cash_balance(prev_app_detail_cash[["SK_ID_PREV"]], pos_cash_balance=pos_cash_balance),
        left_on="SK_ID_PREV",
        right_index=True)
    features_previous_cash["total_months_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').n_months.sum()
    features_previous_cash["min_months_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').n_months.min()
    features_previous_cash["max_months_cash_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').n_months.max()
    features_previous_cash["total_prev_terms_cash"] = prev_app_detail_cash.groupby(
        by='SK_ID_CURR').total_prev_terms.sum()

    features_previous_cash["min_prev_term_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').min_prev_term.min()
    features_previous_cash["max_prev_term_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').max_prev_term.max()

    features_previous_cash["max_remaining_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').max_remaining.max()
    features_previous_cash["number_status_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').number_status.sum()

    features_previous_cash["total_dpd_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').total_dpd.sum()
    features_previous_cash["times_dpd_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').flag_dpd.sum()
    features_previous_cash["times_dpd_def_cash"] = prev_app_detail_cash.groupby(by='SK_ID_CURR').flag_dpd_def.sum()

    features_previous_cash["freq_latest_status_cash"] = mode_column(df=prev_app_detail_cash,
                                                                    col='latest_status',
                                                                    id_col='SK_ID_CURR',
                                                                    cols_count=['SK_ID_PREV'],
                                                                    cols_sum=['n_months']
                                                                    )

    features_previous_cash["flag_cash"] = features_previous_cash["total_months_cash"].apply(lambda x: 1 if x > 0 else 0)

    ##Credit cards
    prev_app_detail_cc = (prev_application[["SK_ID_CURR", "SK_ID_PREV"]
                          ].drop_duplicates().set_index('SK_ID_CURR')
                          )
    prev_app_detail_cc = features_previous_cc.merge(prev_app_detail_cc,
                                                        left_index=True,
                                                        right_index=True, )
    prev_app_detail_cc = prev_app_detail_cc.merge(summarize_credit_card(prev_app_detail_cc[["SK_ID_PREV"]], credit_card = credit_card),
                                                  left_on="SK_ID_PREV",
                                                  right_index=True)

    features_previous_cc["n_months_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["n_months_cc"].sum()

    features_previous_cc["mean_amt_balance_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["mean_amt_balance"].mean()
    features_previous_cc["max_amt_balance_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["max_amt_balance"].max()
    features_previous_cc["min_amt_balance_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["min_amt_balance"].min()
    features_previous_cc["mean_amt_limit_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["mean_amt_limit"].mean()
    features_previous_cc["max_amt_limit_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["max_amt_limit"].max()
    features_previous_cc["min_amt_limit_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["min_amt_limit"].min()
    features_previous_cc["mean_amt_drawing_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_amt_drawing_current"].mean()
    features_previous_cc["max_amt_drawing_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_amt_drawing_current"].max()
    features_previous_cc["min_amt_drawing_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_amt_drawing_current"].min()
    features_previous_cc["mean_amt_drawing_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_amt_drawing_other_current"].mean()
    features_previous_cc["max_amt_drawing_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_amt_drawing_other_current"].max()
    features_previous_cc["min_amt_drawing_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_amt_drawing_other_current"].min()
    features_previous_cc["mean_amt_drawing_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_amt_drawing_pos_current"].mean()
    features_previous_cc["max_amt_drawing_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_amt_drawing_pos_current"].max()
    features_previous_cc["min_amt_drawing_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_amt_drawing_pos_current"].min()
    features_previous_cc["mean_min_installment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_min_installment"].mean()
    features_previous_cc["max_min_installment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_min_installment"].max()
    features_previous_cc["min_min_installment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_min_installment"].min()
    features_previous_cc["mean_payment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["mean_payment"].mean()
    features_previous_cc["max_payment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["max_payment"].max()
    features_previous_cc["min_payment_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["min_payment"].min()
    features_previous_cc["mean_payment_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_payment_total"].mean()
    features_previous_cc["max_payment_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["max_payment_total"].max()
    features_previous_cc["min_payment_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_payment_total"].min()
    features_previous_cc["mean_receivable_principal_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_receivable_principal"].mean()
    features_previous_cc["max_receivable_principal_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_receivable_principal"].max()
    features_previous_cc["min_receivable_principal_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_receivable_principal"].min()
    features_previous_cc["mean_receivable_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["mean_receivable"].mean()
    features_previous_cc["max_receivable_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["max_receivable"].max()
    features_previous_cc["min_receivable_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["min_receivable"].min()
    features_previous_cc["mean_receivable_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_receivable_total"].mean()
    features_previous_cc["max_receivable_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_receivable_total"].max()
    features_previous_cc["min_receivable_total_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_receivable_total"].min()
    features_previous_cc["mean_drawings_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_drawings_current"].mean()
    features_previous_cc["max_drawings_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_drawings_current"].max()
    features_previous_cc["min_drawings_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_drawings_current"].min()
    features_previous_cc["mean_drawings_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_drawings_other_current"].mean()
    features_previous_cc["max_drawings_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_drawings_other_current"].max()
    features_previous_cc["min_drawings_other_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_drawings_other_current"].min()
    features_previous_cc["mean_drawings_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_drawings_pos_current"].mean()
    features_previous_cc["max_drawings_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_drawings_pos_current"].max()
    features_previous_cc["min_drawings_pos_current_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_drawings_pos_current"].min()
    features_previous_cc["mean_installments_mature_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "mean_installments_mature"].mean()
    features_previous_cc["max_installments_mature_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "max_installments_mature"].max()
    features_previous_cc["min_installments_mature_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")[
        "min_installments_mature"].min()
    features_previous_cc["total_dpd_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["total_dpd"].sum()
    features_previous_cc["total_dpd_def_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["total_dpd_def"].sum()
    features_previous_cc["flag_dpd_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["flag_dpd"].sum()
    features_previous_cc["flag_dpd_def_cc"] = prev_app_detail_cc.groupby(by="SK_ID_CURR")["flag_dpd_def"].sum()
    features_previous_cc["latest_status_cc"] = mode_column(df=prev_app_detail_cc,
                                                           col='latest_status',
                                                           id_col='SK_ID_CURR',
                                                           cols_count=['SK_ID_PREV'],
                                                           cols_sum=['max_amt_balance']
                                                           )
    features_previous = pd.concat(
        [features_previous, features_previous_instlmnt, features_previous_cash, features_previous_cc], axis = 1)
    return features_previous


#
def bureau_balance_summary(id_loans,
                           bureau_balance):
    """We are working only with knowns"""
    # Flag if canceled
    bureau_id = id_loans[["SK_ID_BUREAU"]].drop_duplicates().set_index("SK_ID_BUREAU")
    cancelled = bureau_balance[bureau_balance.STATUS == "C"][["SK_ID_BUREAU"]]
    cancelled["FLAG_CANCELLED"] = 1
    cancelled = cancelled.drop_duplicates()
    cancelled.set_index("SK_ID_BUREAU")
    bureau_id["flag_cancelled"] = cancelled.set_index("SK_ID_BUREAU")

    bureau_id["total_months_balance"] = bureau_balance.groupby(by="SK_ID_BUREAU")["MONTHS_BALANCE"].count()

    # Number of times with DPD (Status > 1)
    bureau_balance.loc[bureau_balance["STATUS"].apply(lambda x: x in ['1', '2', '3', '5', '4']), "DPD_FLAG"] = "DPD"
    bureau_id["months_dpd"] = bureau_balance[bureau_balance["STATUS"].apply(lambda x: x in ['1', '2', '3', '5', '4']
                                                                            )].groupby(by="SK_ID_BUREAU")[
        "MONTHS_BALANCE"].count()
    # Number of times with DPD 31+ (Status > 2)
    bureau_id["months_dpd_31"] = bureau_balance[bureau_balance["STATUS"].apply(lambda x: x in ['2', '3', '5', '4']
                                                                               )].groupby(by="SK_ID_BUREAU")[
        "MONTHS_BALANCE"].count()

    # Number of times with DPD 120+ (Status > 5)
    bureau_id["months_dpd_120"] = bureau_balance[bureau_balance["STATUS"].apply(lambda x: x in ['5', ]
                                                                                )].groupby(by="SK_ID_BUREAU")[
        "MONTHS_BALANCE"].count()
    return bureau_id
def get_features_bureau(id_loans,
                        bureau,
                        bureau_balance):
    features_bureau = id_loans[["SK_ID_CURR"]].set_index("SK_ID_CURR")
    features_bureau["n_total_credits"] = bureau.groupby(by="SK_ID_CURR")["SK_ID_BUREAU"].count()
    features_bureau["n_active_credits"] = bureau[bureau["CREDIT_ACTIVE"] == 'Active'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["n_closed_credits"] = bureau[bureau["CREDIT_ACTIVE"] == 'Closed'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["n_sold_credits"] = bureau[bureau["CREDIT_ACTIVE"] == 'Sold'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["n_bad_credits"] = bureau[bureau["CREDIT_ACTIVE"] == 'Bad debt'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["freq_active"] = mode_column(
        df=bureau,
        col='CREDIT_ACTIVE',
        id_col='SK_ID_CURR',
        cols_count=['SK_ID_BUREAU'],
        cols_sum=['AMT_CREDIT_SUM_LIMIT']
    )
    features_bureau["n_credits_currency_1"] = \
    bureau[bureau["CREDIT_CURRENCY"] == 'currency 1'].groupby(by="SK_ID_CURR")["SK_ID_BUREAU"].count()
    features_bureau["n_credits_currency_2"] = \
    bureau[bureau["CREDIT_CURRENCY"] == 'currency 2'].groupby(by="SK_ID_CURR")["SK_ID_BUREAU"].count()
    features_bureau["n_credits_currency_3"] = \
    bureau[bureau["CREDIT_CURRENCY"] == 'currency 3'].groupby(by="SK_ID_CURR")["SK_ID_BUREAU"].count()
    features_bureau["n_credits_currency_4"] = \
    bureau[bureau["CREDIT_CURRENCY"] == 'currency 4'].groupby(by="SK_ID_CURR")["SK_ID_BUREAU"].count()
    features_bureau["freq_currency"] = mode_column(
        df=bureau,
        col='CREDIT_CURRENCY',
        id_col='SK_ID_CURR',
        cols_count=['SK_ID_BUREAU'],
        cols_sum=['AMT_CREDIT_SUM_LIMIT']
    )

    features_bureau["n_credits_consumer"] = bureau[bureau["CREDIT_TYPE"] == 'Consumer credit'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["n_credits_car_loan"] = bureau[bureau["CREDIT_TYPE"] == 'Car loan'].groupby(by = "SK_ID_CURR" )["SK_ID_BUREAU"].count()
    features_bureau["n_credits_credit_card"] = bureau[bureau["CREDIT_TYPE"] == 'Credit card'].groupby(by="SK_ID_CURR")[
        "SK_ID_BUREAU"].count()
    features_bureau["n_credits_other"] = bureau[bureau["CREDIT_TYPE"
                                                ].apply(lambda x: x not in ['Consumer credit',
                                                                'Credit card',
                                                                'Car loan'])
                                                     ].groupby(by="SK_ID_CURR"
                                                               )["SK_ID_BUREAU"].count()
    features_bureau["freq_credit_type"] = mode_column(
        df=bureau,
        col='CREDIT_TYPE',
        id_col='SK_ID_CURR',
        cols_count=['SK_ID_BUREAU'],
        cols_sum=['AMT_CREDIT_SUM_LIMIT']
    )

    features_bureau["min_days_credits"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT"].min()
    features_bureau["max_days_credit"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT"].max()

    features_bureau["min_days_credit_overdue"] = bureau.groupby(by="SK_ID_CURR")["CREDIT_DAY_OVERDUE"].min()
    features_bureau["flag_credit_overdue"] = features_bureau["min_days_credit_overdue"].apply(
        lambda x: 1 if x > 0 else 0)
    features_bureau["max_days_credit_overdue"] = bureau.groupby(by="SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max()

    features_bureau["min_days_duration"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT_ENDDATE"].min()
    features_bureau["max_days_duration"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT_ENDDATE"].max()

    features_bureau["min_days_ended"] = bureau.groupby(by="SK_ID_CURR")["DAYS_ENDDATE_FACT"].min()
    features_bureau["max_days_ended"] = bureau.groupby(by="SK_ID_CURR")["DAYS_ENDDATE_FACT"].max()

    features_bureau["min_amt_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].min()
    features_bureau["max_amt_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].max()

    features_bureau["min_days_update"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT_UPDATE"].min()
    features_bureau["max_days_update"] = bureau.groupby(by="SK_ID_CURR")["DAYS_CREDIT_UPDATE"].max()

    features_bureau["total_prolong"] = bureau.groupby(by="SK_ID_CURR")["CNT_CREDIT_PROLONG"].sum()

    features_bureau["mean_sum_debt"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].mean()
    features_bureau["max_sum_debt"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].max()
    features_bureau["min_sum_debt"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].min()
    features_bureau["median_sum_debt"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].median()

    features_bureau["mean_sum_limit"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].mean()
    features_bureau["max_sum_limit"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].max()
    features_bureau["min_sum_limit"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].min()
    features_bureau["median_sum_limit"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].median()

    features_bureau["mean_sum_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].mean()
    features_bureau["max_sum_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].max()
    features_bureau["min_sum_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].min()
    features_bureau["median_sum_overdue"] = bureau.groupby(by="SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].median()

    features_bureau["mean_annuity"] = bureau.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].mean()
    features_bureau["max_annuity"] = bureau.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].max()
    features_bureau["min_annuity"] = bureau.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].min()
    features_bureau["median_annuity"] = bureau.groupby(by="SK_ID_CURR")["AMT_ANNUITY"].median()

    bureau_details = (bureau[["SK_ID_CURR", "SK_ID_BUREAU"]
                      ].drop_duplicates().set_index('SK_ID_CURR')
                      )
    applications = pd.DataFrame(features_bureau.index)
    bureau_details = applications.merge(bureau_details,
                                                   left_index = True,
                                                   right_index = True)
    bureau_details = bureau_details.merge(bureau_balance_summary(bureau_details[["SK_ID_BUREAU"]], bureau_balance = bureau_balance),
                                          left_on="SK_ID_BUREAU",
                                          right_index=True)

    features_bureau["n_cancelled"] = bureau_details.groupby(by="SK_ID_CURR")["flag_cancelled"].sum()

    features_bureau["tota_months_balance"] = bureau_details.groupby(by="SK_ID_CURR")["total_months_balance"].sum()
    features_bureau["min_months_balance"] = bureau_details.groupby(by="SK_ID_CURR")["total_months_balance"].min()
    features_bureau["max_months_balance"] = bureau_details.groupby(by="SK_ID_CURR")["total_months_balance"].max()

    features_bureau["total_months_dpd"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd"].sum()
    features_bureau["min_months_dpd"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd"].min()
    features_bureau["max_months_dpd"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd"].max()

    features_bureau["total_months_dpd_31"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_31"].sum()
    features_bureau["min_months_dpd_31"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_31"].min()
    features_bureau["max_months_dpd_31"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_31"].max()

    features_bureau["total_months_dpd_120"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_120"].sum()
    features_bureau["min_months_dpd_120"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_120"].min()
    features_bureau["max_months_dpd_120"] = bureau_details.groupby(by="SK_ID_CURR")["months_dpd_120"].max()

    return features_bureau
