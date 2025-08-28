import argparse
import itertools
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Set

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

from kcat_prediction_slim.default_path import DefaultPath

warnings.filterwarnings("ignore")

plt.style.use(DefaultPath().project_root / 'kcat_prediction_slim' / 'CCB_plot_style_0v4.mplstyle')
c_styles = mpl.rcParams['axes.prop_cycle'].by_key()['color']  # fetch the defined color styles
high_contrast = ['#004488', '#DDAA33', '#BB5566', '#000000']

fixed_models = [
    # "structural_fp",
    # "difference_fp",
    "DRFP",
    "ESM1b",
    "ESM1b_ts",
    # "ESM1b_ts_diff_fp",
    # "ESM1b_ts_DRFP",
]

MULTIPLE_LINE_LABEL_SPACE = '     '

fixed_models_drfp_mean = [
    "ESM1b_ts_DRFP_mean"]
model_names = {
    "structural_fp": "str. FP",
    "difference_fp": "diff. FP",
    "ESM1b": "ESM-1b$_\\mathrm{{MEAN}}$",
    "DRFP": "DRFP",
    "ESM1b_ts": "ESM-1b$_\\mathrm{{ESP}}$",
    "ESM1b_ts_diff_fp": "ESM-1b$_\\mathrm{{ESP}}$\n + diff. FP",
    "ESM1b_ts_DRFP": "ESM-1b$_\\mathrm{{ESP}}$\n + DRFP",
    "ESM1b_ts_DRFP_mean": "ESM-1b$_\\mathrm{{ESP}}$     \n + DRFP (mean)",
    #
    # "ESM1B_EnzSRP": "ESM1B EnzSRP",
    # "ESM1B_EnzSRP_DRFP": "ESM1B EnzSRP\n + DRFP",
    # "ESM1B_EnzSRP_kcat_sequence_embeddings_250407_182715": "250407_182715",
}


def load_result(result_dirs, models, seeds):
    model_results = {}
    for i, model in enumerate(models):
        results = []
        for seed in seeds:
            for result_dir in result_dirs:
                try:
                    pred_y = np.load(result_dir / f"y_test_pred_xgboost_{model}_{seed}.npy")
                    test_y = np.load(result_dir / f"y_test_true_xgboost_{model}_{seed}.npy")
                    results.append((pred_y, test_y))
                except Exception as e:
                    pass
        model_results[model] = results
    return model_results


def filtered_unfiltered_indexes(test_y):
    data_test = pd.read_pickle(
        DefaultPath().original_data_dir / "kcat_data" / "splits" / "test_df_kcat.pkl")
    data_test_filtered = pd.read_csv(
        DefaultPath().project_root / "data" / "test_df_kcat_filtered_sequence_ids.csv")

    assert np.all(np.isclose(test_y, np.array(data_test['geomean_kcat']), rtol=1e-5, atol=1e-8)), "Arrays differ"

    unfiltered_id_set = set(data_test_filtered['Sequence ID'])
    match_mask = data_test['Sequence ID'].isin(unfiltered_id_set)
    unfiltered_indices = data_test[match_mask].index.tolist()
    filtered_indices = data_test[~match_mask].index.tolist()
    return filtered_indices, unfiltered_indices


def load_filtered_result(result_dirs, models, seeds):
    results = load_result(result_dirs, models, seeds)
    processed_results = {}
    for i, (model, seed_results) in enumerate(results.items()):
        model_result = []
        for pred_y, test_y in seed_results:
            indexes, _ = filtered_unfiltered_indexes(test_y)
            model_result.append((pred_y[indexes], test_y[indexes]))
        processed_results[model] = model_result
    return processed_results


def load_unfiltered_result(result_dirs, models, seeds):
    results = load_result(result_dirs, models, seeds)
    processed_results = {}
    for i, (model, seed_results) in enumerate(results.items()):
        model_result = []
        for pred_y, test_y in seed_results:
            _, indexes = filtered_unfiltered_indexes(test_y)
            model_result.append((pred_y[indexes], test_y[indexes]))
        processed_results[model] = model_result
    return processed_results


# def clean_label(text: str):
#     temp = text.replace('ESM1B_EnzSRP_kcat_sequence_embeddings', '')
#     temp = temp.replace('ESM1B_EnzSRP_DRFP_kcat_sequence_embeddings', '\n + DRFP (mean)')
#     return temp
def clean_label(text: str):
    if 'ESM1B_EnzSRP_kcat_sequence_embeddings' in text:
        return 'ESM-1b$_\\mathrm{{DA}}$'
    if 'ESM1B_EnzSRP_DRFP_mean_kcat_sequence_embeddings' in text:
        return 'ESM-1b$_\\mathrm{{DA}}$     \n + DRFP (mean)'
    return text


def show_mse(ax, results):
    # ax.set_ylim(0.7, 1.0)
    # ax.set_xlim(0.5, len(models) + 0.5)
    box_data = []
    for i, (model, seed_results) in enumerate(results.items()):
        model_result = [np.mean(abs(test_y - pred_y) ** 2) for pred_y, test_y in seed_results]
        box_data.append(model_result)
    ax.boxplot(box_data)
    ax.set_xticklabels(results.keys(), rotation=45, ha='right', rotation_mode='anchor')
    for label in ax.get_xticklabels():
        if '\n' in label._text:
            label.set_va('center')
    ax.set_ylabel("Mean squared error")

def significance_marker(p):
    return "✅" if p < 0.05 else "❌"

def run_t_test_for_mse(results):
    print()
    print("MSE")
    data = {}
    for model, seed_results in results.items():
        model_result = [np.mean(abs(test_y - pred_y) ** 2) for pred_y, test_y in seed_results]
        print(f"{model}, median {np.median(model_result)}")
        data[model] = model_result
    for (cond1, data1), (cond2, data2) in itertools.combinations(data.items(), 2):
        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
        marker = significance_marker(p_val)
        print(f"{marker} {cond1} vs {cond2}: t = {t_stat:.3f}, p = {p_val:.3g}")


def run_t_test_for_coefficient_of_determination(results):
    print()
    print("R2")
    data = {}
    for model, seed_results in results.items():
        model_result = [r2_score(test_y, pred_y) for pred_y, test_y in seed_results]
        print(f"{model}, median {np.median(model_result)}")
        data[model] = model_result
    for (cond1, data1), (cond2, data2) in itertools.combinations(data.items(), 2):
        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
        marker = significance_marker(p_val)
        print(f"{marker} {cond1} vs {cond2}: t = {t_stat:.3f}, p = {p_val:.3g}")


def show_coefficient_of_determination(ax, results):
    box_data = []
    for i, (model, seed_results) in enumerate(results.items()):
        model_result = [r2_score(test_y, pred_y) for pred_y, test_y in seed_results]
        box_data.append(model_result)
    ax.boxplot(box_data)
    ax.set_xticklabels(results.keys(), rotation=45, ha='right', rotation_mode='anchor')
    for label in ax.get_xticklabels():
        if '\n' in label._text:
            label.set_va('center')

    ax.set_ylabel("Coefficient of determination")


def list_subdirectories(path: Path) -> List[Path]:
    return sorted(
        [p for p in path.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )


class DrawType(Enum):
    ALL = 'all'
    FILTERED = 'filtered'
    UNFILTERED = 'unfiltered'


def main(target_ver: str, x_models: Set[str], x_seeds: Set[int], draw_type: DrawType,
        results_parent_dir: Path,
         model_name_prefix='ESM1B_EnzSRP_kcat_sequence_embeddings',
         model_name_prefix2='ESM1B_EnzSRP_DRFP_mean_kcat_sequence_embeddings', out_file_stem='boxplots'):
    x_models = [*fixed_models,
                *[f"{model_name_prefix}_{n}" for n in x_models],
                *fixed_models_drfp_mean,
                *[f"{model_name_prefix2}_{n}" for n in x_models]]

    out_dir = DefaultPath().build / 'viz' / target_ver
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = list_subdirectories(results_parent_dir / target_ver)
    if draw_type == DrawType.ALL:
        results = load_result(dirs, x_models, x_seeds)
    elif draw_type == DrawType.FILTERED:
        results = load_filtered_result(dirs, x_models, x_seeds)
    elif draw_type == DrawType.UNFILTERED:
        results = load_unfiltered_result(dirs, x_models, x_seeds)
    else:
        raise ValueError('unexpected')
    label_map = {model: model_names[model] if model in model_names.keys() else clean_label(model) for model in x_models}

    results = {label_map[k]: v for k, v in results.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(bottom=0.3)
    show_coefficient_of_determination(axes[0], results)
    run_t_test_for_coefficient_of_determination(results)
    show_mse(axes[1], results)
    run_t_test_for_mse(results)
    plt.savefig(out_dir / f"{out_file_stem}.png")
    plt.savefig(out_dir / f"{out_file_stem}.pdf")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-app-ver', type=str, default='v2_1_0')
    parser.add_argument('--models', nargs='+', default=['250420_121652'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument('--result-parent-dir', type=Path, nargs=None)

    # result_parent_dir = DefaultPath().build / 'server_training_results'

    args = parser.parse_args()
    print('boxplot_all')
    main(target_ver=args.target_app_ver, x_models=args.models, x_seeds=args.seeds, draw_type=DrawType.ALL,
         out_file_stem='boxplot_all', results_parent_dir=args.result_parent_dir)
    print('boxplot_filtered')
    main(target_ver=args.target_app_ver, x_models=args.models, x_seeds=args.seeds, draw_type=DrawType.FILTERED,
         out_file_stem='boxplot_filtered', results_parent_dir=args.result_parent_dir)
    print('boxplot_unfiltered')
    main(target_ver=args.target_app_ver, x_models=args.models, x_seeds=args.seeds, draw_type=DrawType.UNFILTERED,
         out_file_stem='boxplot_unfiltered', results_parent_dir=args.result_parent_dir)
