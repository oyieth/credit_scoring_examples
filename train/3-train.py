"""Step 3 - train the classifier and write model, metrics, plots and scored data.

Reads  : models/<name>/data/{train,test}_<name>_preprocessed.csv, preprocessor artifact
Writes : models/<name>/artifacts/model<name>.joblib.dat, metrics json, img/*, scored csvs; appends to log.txt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from src.model_paths import LOG_PATH, PIPELINE_MODEL, ReadPathsPreprocess, ReadPathsTrain, WritePathsTrain
from src.preprocessor_utils import get_column_names
from src.train import performance_metrics, save_predicted_scores, train_model
from src.utils import is_jsonable


def train(
        model_name,
        target_col,
        random_state,
        model_type,
        use_best_params,
        param_search,
        distributions,
        scoring,
        n_splits,
        n_iter,
        timeout,
        n_trials,
        manual_threshold=0.3,
        params_file=None,
):
    preprocess_path = ReadPathsPreprocess(model_name)
    train_w_path = WritePathsTrain(model_name)
    train_r_path = ReadPathsTrain(model_name)
    if params_file:                      # hyperparameters tuned elsewhere (e.g. notebook 3.2)
        train_r_path.metrics_file = Path(params_file)
        use_best_params = True

    train_df = pd.read_csv(preprocess_path.train_dataset_path)
    test_df = pd.read_csv(preprocess_path.test_dataset_path)
    preprocessor = joblib.load(preprocess_path.preprocessor_path)
    columns_model = get_column_names(preprocessor)
    x_train, y_train = train_df[columns_model], train_df[target_col]
    x_test, y_test = test_df[columns_model], test_df[target_col]

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        model_type=model_type,
        read_paths=train_r_path,
        use_best_params=use_best_params,
        param_search=param_search,
        random_state=random_state,
        distributions=distributions,
        scoring=scoring,
        n_splits=n_splits,
        n_iter=n_iter,
        w_suffix=model_name + "_" + model_type,
        write_paths=train_w_path,
        timeout=timeout,
        n_trials=n_trials,
    )
    print(f"Classifier Train score: {model.score(x_train, y_train)}")
    print(f"Classifier Test score: {model.score(x_test, y_test)}\n")

    joblib.dump(model, train_w_path.model_path)
    print(f"Model saved at {train_w_path.model_path}\n")

    save_predicted_scores(model, x_test, test_df, x_train, train_df, train_w_path)
    metric_scores = performance_metrics(
        x_train, y_train, x_test, y_test, model, train_w_path, manual_threshold, x_test, y_test
    )
    metric_scores["params"] = {
        k: v for k, v in model.__dict__.items() if is_jsonable(v) and k[0] != "_"
    }

    with open(train_w_path.metrics_file, "w", encoding="utf-8") as fd:
        json.dump(metric_scores, fd, indent=4)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n{train_w_path.metrics_file}; \"{metric_scores}\"")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=PIPELINE_MODEL)
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-type", default="xgb", choices=["xgb", "cat", "lgb"])
    parser.add_argument("--use-best-params", action="store_true",
                        help="reuse hyperparameters stored in the model's metrics json")
    parser.add_argument("--params-file", type=Path, default=None,
                        help="json with tuned hyperparameters (e.g. best_params<name>.json); implies --use-best-params")
    parser.add_argument("--param-search", default="randomized_search",
                        choices=["randomized_search", "optuna", "oot_optuna"])
    parser.add_argument("--distributions", default="xgb2")
    parser.add_argument("--scoring", default="neg_log_loss")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--n-trials", type=int, default=200)
    args = parser.parse_args(argv)
    train(
        model_name=args.model_name, target_col=args.target, random_state=args.random_state,
        model_type=args.model_type, use_best_params=args.use_best_params,
        params_file=args.params_file, param_search=args.param_search, distributions=args.distributions, scoring=args.scoring,
        n_splits=args.n_splits, n_iter=args.n_iter, timeout=args.timeout, n_trials=args.n_trials,
    )


if __name__ == "__main__":
    raise SystemExit(main())
