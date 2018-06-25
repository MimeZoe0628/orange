import lightgbm as lightgbm_ensemble

from Orange.base import SklLearner
from Orange.classification.base_classification import (
    SklLearnerClassification, SklModelClassification
)
from Orange.regression.base_regression import (
    SklLearnerRegression, SklModelRegression
)

__all__ = ['SklLightGBMClassificationLearner', 'SklLightGBMRegressionLearner']


class SklLightGBMClassifier(SklModelClassification):
    pass


class SklLightGBMClassificationLearner(SklLearnerClassification):
    __wraps__ = lightgbm_ensemble.LGBMClassifier
    __returns__ = SklLightGBMClassifier

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, sub_sample=1.0,
                 subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklLightGBMRegressor(SklModelRegression):
    pass


class SklLightGBMRegressionLearner(SklLearnerRegression):
    __wraps__ = lightgbm_ensemble.LGBMRegressor
    __returns__ = SklLightGBMRegressor

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, sub_sample=1.0,
                 subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
