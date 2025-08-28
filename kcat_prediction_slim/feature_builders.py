from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


class FeatureEnum(Enum):
    ESM1B = 'ESM1b'
    ESM1B_TS = 'ESM1b_ts'  # ESM1b_ESP
    DRFP = 'DRFP'
    # DIFFERENCE_FP = 'difference_fp'
    # STRUCTURAL_FP = 'structural_fp'
    # ESM1B_TS_DIFF_FP = 'ESM1b_ts_diff_fp'
    # ESM1B_TS_DRFP = 'ESM1b_ts_DRFP'
    ESM1B_TS_DRFP_MEAN = 'ESM1b_ts_DRFP_mean'  # ESM1b_ESP + DRFP
    #
    ESM1B_ENZSRP = 'ESM1B_EnzSRP'  # ESM1B_DA
    # ESM1B_ENZSRP_DRFP = 'ESM1B_EnzSRP_DRFP'
    ESM1B_ENZSRP_DRFP_MEAN = 'ESM1B_EnzSRP_DRFP_mean'  # ESM1b_DA + DRFP


def get_key(feature: FeatureEnum, embedding_file: Optional[Path] = None):
    return feature.value + '_' + embedding_file.stem if embedding_file else feature.value


def build_simple_feature(data, target: FeatureEnum):
    return np.array(list(data[target.value]))


# def build_structual_fp(data, target: FeatureEnum):
#     assert target == FeatureEnum.STRUCTURAL_FP
#     feat = ()
#     for ind in data.index:
#         feat = feat + (np.array(list(data[target.value][ind])).astype(int),)
#     return np.array(feat)


# def build_concat_drfp_fp(data, target: FeatureEnum):
#     assert target == FeatureEnum.ESM1B_TS_DRFP or target == FeatureEnum.ESM1B_ENZSRP_DRFP
#     feat = np.array(list(data["DRFP"]))
#     if target == FeatureEnum.ESM1B_TS_DRFP:
#         return np.concatenate([feat, np.array(list(data["ESM1b_ts"]))], axis=1)
#     elif target == FeatureEnum.ESM1B_ENZSRP_DRFP:
#         return np.concatenate([feat, np.array(list(data["ESM1B_EnzSRP"]))], axis=1)


# def build_esm1b_ts_diff_fp(data, target: FeatureEnum):
#     assert target == FeatureEnum.ESM1B_TS_DIFF_FP
#     feat = np.array(list(data["difference_fp"]))
#     return np.concatenate([feat, np.array(list(data["ESM1b_ts"]))], axis=1)


def get_feature_func(target: FeatureEnum) -> Callable[[pd.DataFrame, FeatureEnum], np.array]:
    if target in [FeatureEnum.ESM1B, FeatureEnum.ESM1B_TS, FeatureEnum.DRFP,
                  # FeatureEnum.DIFFERENCE_FP,
                  FeatureEnum.ESM1B_ENZSRP]:
        return build_simple_feature
    # elif target == FeatureEnum.STRUCTURAL_FP:
    #     return build_structual_fp
    # elif target == FeatureEnum.ESM1B_TS_DRFP or target == FeatureEnum.ESM1B_ENZSRP_DRFP:
    #     return build_concat_drfp_fp
    # elif target == FeatureEnum.ESM1B_TS_DIFF_FP:
    #     return build_esm1b_ts_diff_fp
    raise ValueError('Undefined')
