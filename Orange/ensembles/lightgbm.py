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

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', random_state=None, preprocessors=None):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(
                base_estimator.CLASSIFICATION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklLightGBMRegressor(SklModelRegression):
    pass


class SklLightGBMRegressionLearner(SklLearnerRegression):
    __wraps__ = lightgbm_ensemble.LGBMRegressor
    __returns__ = SklLightGBMRegressor

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 loss='linear', random_state=None, preprocessors=None):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(
                base_estimator.REGRESSION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
