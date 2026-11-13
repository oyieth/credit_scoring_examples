"""Step 2 - fit the preprocessing pipeline on the train set and apply it to train and test.

Reads  : data/application_{train,test}_all_features.csv (from step 1), schema/data_schema.json
Writes : models/<name>/artifacts/preprocessor_<name>.joblib.dat,
         models/<name>/data/{train,test}_<name>_preprocessed.csv, encoder dict, schema/data_schema_train.json

Both files need the target column: step 3 evaluates on the test set, so it must be labeled.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.model_paths import PIPELINE_MODEL, SCHEMA_DIR, ExtractPaths, WritePathsPreprocess
from src.preprocessor_utils import get_preprocessor, save_preprocessed_data


def preprocessor(model_name, target_col, other_metrics, data_schema_path):
    paths = ExtractPaths()
    write_paths = WritePathsPreprocess(model_name)

    train = pd.read_csv(paths.train_all_features_path)
    test = pd.read_csv(paths.test_all_features_path)
    for name, data in (("train", train), ("test", test)):
        if target_col not in data.columns:
            raise ValueError(
                f"The {name} set has no `{target_col}` column. Step 3 evaluates on the test set, "
                "so it needs labels (the original Kaggle test file has none)."
            )

    # Fit on train only
    fitted, column_names = get_preprocessor(
        x=train,
        y=train[target_col],
        data_schema_path=data_schema_path,
        write_paths=write_paths,
        other_metrics=other_metrics,
        target_col=target_col,
    )

    for data, path in ((train, write_paths.train_dataset_path), (test, write_paths.test_dataset_path)):
        save_preprocessed_data(
            x=data,
            y=data[target_col],
            preprocessor=fitted,
            column_names=column_names,
            other_metrics=other_metrics,
            target=target_col,
            path=path,
            save=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=PIPELINE_MODEL)
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--other-metrics", nargs="*", default=["SK_ID_CURR"],
                        help="ID / non-feature columns carried through to the output")
    parser.add_argument("--schema", type=Path, default=SCHEMA_DIR / "data_schema.json")
    args = parser.parse_args(argv)
    preprocessor(args.model_name, args.target, args.other_metrics, args.schema)


if __name__ == "__main__":
    raise SystemExit(main())
