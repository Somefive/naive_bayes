class NaiveBayesFeature(object):

    def __init__(self, feature_name, feature_type):
        self.name = feature_name
        self.type = feature_type

    def __str__(self):
        return '%s[%s]:\t' % (self.name, self.type)

    def add(self, value):
        pass

    def log_prob(self, value):
        return 0
