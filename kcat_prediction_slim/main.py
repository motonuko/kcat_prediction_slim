import argparse
import ast
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from kcat_prediction_slim.default_path import DefaultPath
from kcat_prediction_slim.feature_builders import FeatureEnum, get_feature_func, get_key
from kcat_prediction_slim.hyperparameter import run_hyperparameter_optimization

# warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb

import matplotlib as mpl

from kcat_prediction_slim.package_version import get_package_version

# plt.style.use('CCB_plot_style_0v4.mplstyle')
c_styles = mpl.rcParams['axes.prop_cycle'].by_key()['color']  # fetch the defined color styles
high_contrast = ['#004488', '#DDAA33', '#BB5566', '#000000']

current_time_str = datetime.now().strftime("%y%m%d_%H%M%S")
out_dir = DefaultPath().build / 'training_results' / get_package_version() / current_time_str
out_dir.mkdir(parents=True, exist_ok=True)


def load_our_csv(csv_file):
    dfx = pd.read_csv(csv_file)
    dfx["embedding"] = dfx["embedding"].apply(ast.literal_eval)
    dfx = dfx.rename(columns={'sequence': 'trimmed_sequence'})
    dfx = dfx.rename(columns={'embedding': FeatureEnum.ESM1B_ENZSRP.value})
    original_len = len(dfx)
    dfx = dfx.drop_duplicates(subset='trimmed_sequence', keep='first')
    if original_len != len(dfx):
        warnings.warn('duplicated sequence(s)')
    return dfx


def _loading_data():
    data_train = pd.read_pickle(DefaultPath().original_data_dir / "kcat_data" / "splits" / "train_df_kcat.pkl")

    data_test = pd.read_pickle(DefaultPath().original_data_dir / "kcat_data" / "splits" / "test_df_kcat.pkl")

    data_train.rename(columns={"geomean_kcat": "log10_kcat"}, inplace=True)
    data_test.rename(columns={"geomean_kcat": "log10_kcat"}, inplace=True)
    # len(data_train), len(data_test)

    train_indices = list(
        np.load(DefaultPath().original_data_dir / "kcat_data" / "splits" / "CV_train_indices.npy", allow_pickle=True))
    test_indices = list(
        np.load(DefaultPath().original_data_dir / "kcat_data" / "splits" / "CV_test_indices.npy", allow_pickle=True))
    return data_train, data_test, train_indices, test_indices


def loading_data():
    return _loading_data()


def loading_data_with_enzsrp_embedding(embedding_file: Path):
    data_train, data_test, train_indices, test_indices = _loading_data()

    # Added
    original_data_train_len, original_data_test_len = len(data_train), len(data_test)
    data_train['trimmed_sequence'] = data_train['Sequence'].apply(lambda x: x[:1020])
    data_test['trimmed_sequence'] = data_test['Sequence'].apply(lambda x: x[:1020])
    df_our = load_our_csv(embedding_file)
    data_train = data_train.merge(df_our, how='left', on='trimmed_sequence')
    data_test = data_test.merge(df_our, how='left', on='trimmed_sequence')

    assert data_train[FeatureEnum.ESM1B_ENZSRP.value].notna().all(), "Error: The 'EnzSRP' column contains None (or NaN) values."
    assert data_test[FeatureEnum.ESM1B_ENZSRP.value].notna().all(), "Error: The 'EnzSRP' column contains None (or NaN) values."
    assert len(data_train) == original_data_train_len, (len(data_train), original_data_train_len)
    assert len(data_test) == original_data_test_len
    return data_train, data_test, train_indices, test_indices


def train_and_eval(original_param, target: FeatureEnum, train_indices, test_indices, data_train, data_test,
                   build_feature_func, key=''):
    train_X = build_feature_func(data_train, target)
    train_Y = np.array(list(data_train["log10_kcat"]))
    test_X = build_feature_func(data_test, target)
    test_Y = np.array(list(data_test["log10_kcat"]))

    key_suffix = f"_{key}" if key else ""

    param = original_param.copy()

    num_round = param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    del param["num_rounds"]

    R2 = []
    MSE = []
    Pearson = []
    y_valid_preds = []

    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
        dvalid = xgb.DMatrix(train_X[test_index])

        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

        y_valid_pred = bst.predict(dvalid)
        y_valid_preds.append(y_valid_pred)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2))
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
        Pearson.append(stats.pearsonr(np.reshape(train_Y[test_index], (-1)), y_valid_pred)[0])

    print(Pearson)
    print(MSE)
    print(R2)

    np.save(out_dir / f"Pearson_CV_xgboost_{target.value}{key_suffix}.npy", np.array(Pearson))
    np.save(out_dir / f"MSE_CV_xgboost_{target.value}{key_suffix}.npy", np.array(MSE))
    np.save(out_dir / f"R2_CV_xgboost_{target.value}{key_suffix}.npy", np.array(R2))

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)

    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    y_test_pred = bst.predict(dtest)
    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred) ** 2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

    print(np.round(Pearson[0], 3), np.round(MSE_dif_fp_test, 3), np.round(R2_dif_fp_test, 3))

    np.save(out_dir / f"y_test_pred_xgboost_{target.value}{key_suffix}.npy", bst.predict(dtest))
    np.save(out_dir / f"y_test_true_xgboost_{target.value}{key_suffix}.npy", test_Y)

    return y_valid_preds, y_test_pred


def full_pipeline(target: FeatureEnum, seed: int):
    assert target != FeatureEnum.ESM1B_TS_DRFP_MEAN
    assert target != FeatureEnum.ESM1B_ENZSRP
    data_train, data_test, train_indices, test_indices = loading_data()
    build_feature_func = get_feature_func(target)
    best_params = run_hyperparameter_optimization(train_indices, test_indices, data_train, target, build_feature_func,
                                                  seed)
    y_valid_preds, y_test_pred = train_and_eval(best_params, target, train_indices, test_indices, data_train, data_test,
                                                build_feature_func, key=f"{seed}")
    return y_valid_preds, y_test_pred


def full_pipeline_for_enzsrp_model(target: FeatureEnum, embedding_file: Path, seed):
    assert target in [FeatureEnum.ESM1B_ENZSRP,
                      # FeatureEnum.ESM1B_ENZSRP_DRFP
                      ], target
    key = embedding_file.stem + f"_{seed}"
    data_train, data_test, train_indices, test_indices = loading_data_with_enzsrp_embedding(embedding_file)
    build_feature_func = get_feature_func(target)
    best_params = run_hyperparameter_optimization(train_indices, test_indices, data_train, target, build_feature_func,
                                                  seed)
    y_valid_preds, y_test_pred = train_and_eval(best_params, target, train_indices, test_indices, data_train, data_test,
                                                build_feature_func, key)
    return y_valid_preds, y_test_pred
    # production_mode(best_params, target)


# for ensemble model
def eval_only_pipeline(target: FeatureEnum, y_valid_preds_dict, y_test_pred_dict, seed: int,
                       embedding_file: Optional[Path] = None):
    assert target in [FeatureEnum.ESM1B_TS_DRFP_MEAN, FeatureEnum.ESM1B_ENZSRP_DRFP_MEAN], target
    if target == FeatureEnum.ESM1B_ENZSRP_DRFP_MEAN:
        assert embedding_file is not None

    data_train, data_test, train_indices, test_indices = loading_data()

    y_valid_pred_DRFP = y_valid_preds_dict[get_key(FeatureEnum.DRFP)]
    y_test_pred_drfp = y_test_pred_dict[get_key(FeatureEnum.DRFP)]

    if target == FeatureEnum.ESM1B_TS_DRFP_MEAN:
        y_valid_pred_esm1b_ts = y_valid_preds_dict[get_key(FeatureEnum.ESM1B_TS)]
        y_test_pred_esm1b_ts = y_test_pred_dict[get_key(FeatureEnum.ESM1B_TS)]
        key = get_key(target, None)
    elif target == FeatureEnum.ESM1B_ENZSRP_DRFP_MEAN:
        y_valid_pred_esm1b_ts = y_valid_preds_dict[get_key(FeatureEnum.ESM1B_ENZSRP, embedding_file)]
        y_test_pred_esm1b_ts = y_test_pred_dict[get_key(FeatureEnum.ESM1B_ENZSRP, embedding_file)]
        key = get_key(target, embedding_file)
    else:
        raise ValueError(f"Unexpected target feature: {target}")

    train_Y = np.array(list(data_train["log10_kcat"]))
    test_Y = np.array(list(data_test["log10_kcat"]))

    R2 = []
    MSE = []
    Pearson = []

    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        y_valid_pred = np.mean([y_valid_pred_DRFP[i], y_valid_pred_esm1b_ts[i]], axis=0)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2))
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
        Pearson.append(stats.pearsonr(np.reshape(train_Y[test_index], (-1)), y_valid_pred)[0])

    print(Pearson)
    print(MSE)
    print(R2)


    np.save(out_dir / f"Pearson_CV_xgboost_{key}_{seed}.npy", np.array(Pearson))
    np.save(out_dir / f"MSE_CV_xgboost_{key}_{seed}.npy", np.array(MSE))
    np.save(out_dir / f"R2_CV_xgboost_{key}_{seed}.npy", np.array(R2))

    y_test_pred = np.mean([y_test_pred_drfp, y_test_pred_esm1b_ts], axis=0)

    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred) ** 2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)
    print(np.round(Pearson[0], 3), np.round(MSE_dif_fp_test, 3), np.round(R2_dif_fp_test, 3))

    np.save(out_dir / f"y_test_pred_xgboost_{key}_{seed}.npy", y_test_pred)
    np.save(out_dir / f"y_test_true_xgboost_{key}_{seed}.npy", test_Y)

    return [], y_test_pred


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_all_patterns(embedding_files: List[Path], run_all: bool = False):
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    features = FeatureEnum if run_all else [FeatureEnum.DRFP,
                                            FeatureEnum.ESM1B_ENZSRP,
                                            # FeatureEnum.ESM1B_ENZSRP_DRFP,
                                            FeatureEnum.ESM1B_ENZSRP_DRFP_MEAN]
    for seed in seeds:
        set_random_seed(seed)
        y_valid_preds_dict = {}
        y_test_pred_dict = {}
        for feature in features:
            print('current feature: ', feature, 'seed: ', seed)
            if feature == FeatureEnum.ESM1B_TS_DRFP_MEAN or feature == FeatureEnum.ESM1B_ENZSRP_DRFP_MEAN:
                # NOTE: This should be called last (it depends on the previous calculation results)
                for embedding_file in embedding_files:
                    eval_only_pipeline(feature, y_valid_preds_dict, y_test_pred_dict, seed=seed,
                                       embedding_file=embedding_file)
            # elif feature == FeatureEnum.ESM1B_ENZSRP or feature == FeatureEnum.ESM1B_ENZSRP_DRFP:
            elif feature == FeatureEnum.ESM1B_ENZSRP:
                assert len(embedding_files) != 0, f"Embedding files should be provided for ENZSRP features."
                for embedding_file in embedding_files:
                    key = get_key(feature, embedding_file)
                    y_valid_preds_dict[key], y_test_pred_dict[key] = full_pipeline_for_enzsrp_model(feature,
                                                                                                    embedding_file=embedding_file,
                                                                                                    seed=seed)
            else:
                key = get_key(feature)
                y_valid_preds_dict[key], y_test_pred_dict[key] = full_pipeline(feature, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Script that accepts file paths as arguments")
    parser.add_argument("embedding_files", type=Path, nargs='*',
                        help="Paths to embedding files or directories to be processed")
    parser.add_argument("--run-all", action="store_true", help="")
    args = parser.parse_args()
    for path in args.embedding_files:
        assert path.exists(), f'Embedding file path does not exist: {path}'
    run_all_patterns(args.embedding_files, run_all=args.run_all)


if __name__ == '__main__':
    main()
