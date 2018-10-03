import math
from naive_bayes_feature import *

FEATURE_TYPE_CATEGORICAL = "Categorical"


class NaiveBayesCategoricalFeature(NaiveBayesFeature):

    def __init__(self, feature_name, category_list=None,
                 increase_category_enable=True, smooth=False):
        """
        Build a categorical feature.

        :param feature_name the name of this feature
        :param category_list the categories of this feature
        :param increase_category_enable for values that are not in category_list,
        add them into category list
        :param smooth if using smooth to handle probability
        """
        super().__init__(feature_name, FEATURE_TYPE_CATEGORICAL)
        self.total = 0
        self.counter = dict()
        self.category_list = category_list if category_list else list()
        self.increase_category_enable = increase_category_enable
        self.smooth = smooth
        for category in category_list:
            self.counter[category] = 0

    def __str__(self):
        return super().__str__() + \
            ', '.join(['%s:%.4f' % (category, self.frequency(category))
                       for category in self.category_list])

    def add(self, value):
        if value in self.counter:
            self.counter[value] += 1
            self.total += 1
        elif self.increase_category_enable:
            self.category_list.append(value)
            self.counter[value] = 1
            self.total += 1

    def frequency(self, value):
        v = self.counter[value] if value in self.counter else 0
        return (v + 1) / (self.total + len(self.category_list)) \
            if self.smooth else v / self.total

    def log_prob(self, value):
        try:
            return math.log2(self.frequency(value))
        except ValueError:
            return -math.inf
