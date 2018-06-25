from Orange.base import SklModel
from Orange.ensembles import (
    SklLightGBMClassificationLearner, SklLightGBMRegressionLearner
)
from Orange.data import Variable
from Orange.modelling import SklFitter
from Orange.preprocess.score import LearnerScorer

__all__ = ['LightGBMLearner']


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = Variable

    def score(self, data):
        model = self.get_learner(data)(data)
        return model.skl_model.feature_importances_


# 随机森林学习器，包括分类、回归实现
class LightGBMLearner(SklFitter, _FeatureScorerMixin):
    name = 'LighGBM'

    __fits__ = {'classification': SklLightGBMClassificationLearner,
                'regression': SklLightGBMRegressionLearner}

    __returns__ = SklModel
