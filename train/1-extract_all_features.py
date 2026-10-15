"""Step 1 - build the model-ready feature tables from the raw Kaggle tables.

Reads  : data/application_{train,test}.csv + the relational tables
Writes : data/application_{train,test}_all_features.csv  (SK_ID_CURR kept as first column)
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))        # so `python train/1-...py` finds `src` from any cwd

import pandas as pd

from src.feature_extraction import get_previous_application_features, get_features_bureau
from src.model_paths import ExtractPaths

ID_COL = "SK_ID_CURR"


def build_all_features(train_raw, test_raw, paths):
    """Application tables + bureau + previous-application features, as (train, test).

    Features are aggregates per SK_ID_CURR, so they are computed once for train and
    test together (each big table is read once).
    """
    if set(train_raw[ID_COL]) & set(test_raw[ID_COL]):
        raise ValueError("Train and test applications share SK_ID_CURR values.")
    id_loans = pd.concat([train_raw[[ID_COL]], test_raw[[ID_COL]]], ignore_index=True)

    previous = get_previous_application_features(
        id_loans=id_loans,
        prev_application=pd.read_csv(paths.previous_application_path),
        installments_payments=pd.read_csv(paths.installments_payments_path),
        pos_cash_balance=pd.read_csv(paths.pos_cash_balance_path),
        credit_card=pd.read_csv(paths.credit_card_balance_path),
    )
    bureau = get_features_bureau(
        id_loans=id_loans,
        bureau=pd.read_csv(paths.bureau_path),
        bureau_balance=pd.read_csv(paths.bureau_balance_path),
    )

    out = []
    for raw in (train_raw, test_raw):
        final = raw.set_index(ID_COL).join([bureau, previous], how="left")   # raises on duplicate columns
        if len(final) != len(raw):
            raise ValueError("Row count changed while joining features.")
        out.append(final.reset_index())
    return tuple(out)


def main():
    paths = ExtractPaths()
    train_raw = pd.read_csv(paths.application_train_path)
    test_raw = pd.read_csv(paths.application_test_path)

    train_final, test_final = build_all_features(train_raw, test_raw, paths)

    train_final.to_csv(paths.train_all_features_path, index=False)
    print(f"Train data saved at {paths.train_all_features_path} {train_final.shape}")
    test_final.to_csv(paths.test_all_features_path, index=False)
    print(f"Test data saved at {paths.test_all_features_path} {test_final.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
