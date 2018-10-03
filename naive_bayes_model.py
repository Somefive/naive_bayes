class NaiveBayesModel(object):

    def __init__(self, category_features, categories):
        """
        Build a Naive Bayes Model

        :param category_features: A dictionary with category_name as key and a 
        list of features as value
        :type category_features: dict[str, list(NaiveBayesFeature)]
        :param categories: Output feature
        :type categories: NaiveBayesFeature
        """
        self.category_features = category_features  # type: dict
        self.category_names = list(self.category_features.keys())  # type: list
        self.category_count = len(category_features)
        if self.category_count == 0:
            raise Exception('NaiveBayesModel',
                            'Category number cannot be zero.')
        self.feature_count = len(list(category_features.values())[0])
        if self.feature_count == 0:
            raise Exception('NaiveBayesModel',
                            'Feature number cannot be zero.')
        for category_name, features in category_features.items():
            if len(features) != self.feature_count:
                raise Exception('NaiveBayesModel',
                                'Feature number inconsistent. Should be %d, feature[%s] is %d.'
                                % (self.feature_count, category_name, len(features)))
        self.categories = categories
        self.total = 0

    def print(self):
        print('NaiveBayesModel Trained:%d' % self.total)
        print('\t%s' % self.categories)
        for category in self.category_names:
            print('\tCategory: %s' % category)
            for feature in self.category_features[category]:
                print('\t\t%s' % feature)

    def fit(self, dataset):
        for data in dataset:
            category = data[self.feature_count]
            if category not in self.category_names:
                continue
            self.total += 1
            for i in range(self.feature_count):
                self.category_features[category][i].add(data[i])
            self.categories.add(category)

    def predict(self, data):
        category_id, log_probs = 0, []
        for category in self.category_names:
            l = self.categories.log_prob(category)
            for i in range(self.feature_count):
                l += self.category_features[category][i].log_prob(data[i])
            log_probs.append(l)
            if log_probs[-1] > log_probs[category_id]:
                category_id = len(log_probs) - 1
        return self.category_names[category_id], log_probs

    def evaluate(self, dataset, verbose=True):
        t, cnt = 0, 0
        if verbose:
            from tqdm import tqdm
            dataset = tqdm(dataset)
        for data in dataset:
            if self.predict(data)[0] == data[self.feature_count]:
                t += 1
            cnt += 1
        return t / cnt
