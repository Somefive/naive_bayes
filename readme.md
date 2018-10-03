# Naive Bayes Model
#### Author: Somefive Date: Oct. 2018
This is an implementation of Naive Bayes Model. It supports two different types features: Numeric and Categorical. For numeric ones, currently only Gaussian Distribution is implemented but other distribution could be implemented similarly. For categorical ones, the model will count the occurrence of the specific value of one feature under some category and use it to predict.

Generally speaking,
$P(Y=y|X=x)\propto P(Y=y)P(X=x|Y=y)$, under Bayes Assumption which means different features are independent under the condition of label, $P(X=x|Y=y)=\prod_{i=1}^nP(X_i=x_i|Y=y)$.
Using Maximum Likelihood Estimation we found that, 
$\hat{P}(Y=y)=\frac{\{\#l: Y^{(l)}=y\}}{L}$,$\hat{P}(X_i=x_i|Y_i=y_i)=\frac{\{\#l: X_i^{(l)}=x_i, Y^{(l)}=y\}}{\{\#l: Y^{(l)}=y\}}$. Using this we could build Naive Bayes Model and estimate.

The model could be used as follow
```python
from naive_bayes_model import *
from naive_bayes_gaussian_distribution_feature import NaiveBayesGaussianDistributionFeature
from naive_bayes_categorical_feature import NaiveBayesCategoricalFeature
categories = NaiveBayesCategoricalFeature(
    feature_name=categories_names,
    category_list=categories_list, smooth=False)
category_features = dict()
for category in categories_names:
    category_features[category] = list()
    for i in range(feature_count):
        if IsNumeric(i):
            category_features[category].append(
                NaiveBayesGaussianDistributionFeature(attributes_names[i]))
        else:
            category_features[category].append(
                NaiveBayesCategoricalFeature(feature_name=attributes_names[i],
                                             category_list=attributes[i], smooth=False))
model = NaiveBayesModel(category_features, categories)
model.fit(TrainData)
model.print()
print("TrainData Accuracy: %.4f" % model.evaluate(TrainData))
print("TestData Accuracy: %.4f" % model.evaluate(TestData))
```