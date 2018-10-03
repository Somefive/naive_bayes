import math
from naive_bayes_feature import *

FEATURE_TYPE_GAUSSIAN_DISTRIBUTION = "Gaussian Distribution"


class NaiveBayesGaussianDistributionFeature(NaiveBayesFeature):

    def __init__(self, feature_name):
        """
        Build a numerical feature using gaussian distribution.

        :param feature_name the name of this feature
        """
        super().__init__(feature_name, FEATURE_TYPE_GAUSSIAN_DISTRIBUTION)
        self.data = []
        self.dirty = False
        self.mean = 0
        self.var = 1e-9

    def __str__(self):
        if self.dirty:
            self._update()
        return super().__str__() + 'mean: %.4f, var: %.4f' % (self.mean, self.var)

    def add(self, value):
        self.data.append(value)
        self.dirty = True

    def _update(self):
        self.mean = sum(self.data) / len(self.data)
        self.var = sum([(x - self.mean) ** 2 for x in self.data]) \
            / len(self.data) + 1e-9
        self.dirty = False

    def log_prob(self, value):
        if self.dirty:
            self._update()
        try:
            return math.log2(math.exp(-(value-self.mean)**2/2/self.var) /
                             math.sqrt(2*math.pi*self.var))
        except ValueError:
            return -math.inf
