import warnings

import numpy as np

from kcat_prediction_slim.feature_builders import FeatureEnum

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
import xgboost as xgb

from hyperopt import fmin, rand, hp, Trials


def cross_validation_mse_gradient_boosting(param, train_indices, test_indices, train_X, train_Y, seed: int):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    # param["tree_method"] = "hist"
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param["seed"] = seed

    MSE = []
    R2 = []
    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
        dvalid = xgb.DMatrix(train_X[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2))
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    return (-np.mean(R2))


space_gradient_boosting = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    "max_depth": hp.uniform("max_depth", 4, 12),
    # "subsample": hp.uniform("subsample", 0.7, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds": hp.uniform("num_rounds", 20, 200)}

max_evals = 200  # original
# max_evals = 1  # for debugging


def run_hyperparameter_optimization(train_indices, test_indices, data_train, target: FeatureEnum, build_feature_func,
                                    seed):
    train_X = build_feature_func(data_train, target)
    train_Y = np.array(list(data_train["log10_kcat"]))

    trials = Trials()
    rstate = np.random.RandomState(seed)
    best = fmin(
        fn=lambda params: cross_validation_mse_gradient_boosting(params, train_indices, test_indices, train_X, train_Y,
                                                                 seed),
        space=space_gradient_boosting,
        algo=rand.suggest, max_evals=max_evals, trials=trials, rstate=rstate)
    return best
