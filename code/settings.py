import pandas as pd
import numpy as np
# import out grid search module
from sklearn.model_selection import GridSearchCV

# Import four machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# we will set a random seed to ensure that whenever we use random numbers 
# which is a good amount, we will achieve the same random numbers
np.random.seed(123)

#------------------------------ SET Params --------------------#

# Set up some parameters for our grid search
# We will start with four different machine learning model parameters

# Logistic Regression
lr_params = {'C': [1e-1, 1e0, 1e1, 1e2], 'penalty': ['l1', 'l2']}

# KNN
knn_params = {'n_neighbors': [1, 3, 5, 7]}

# Decision Tree
tree_params = {'max_depth': [None, 1, 3, 5, 7]}

# Random Forest
forest_params = {'n_estimators': [
    10, 50, 100], 'max_depth': [None, 1, 3, 5, 7]}

#-------------------------------------------------------------#


def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model, # the model to grid search
                        params, # the parameter set to try 
                        error_score=0.) # if a parameter set raises an error, continue and set the performance as a big, fat 0
    grid.fit(X, y) # fit the model and parameters
    # our classical metric for performance
    print "Best Accuracy: {}".format(grid.best_score_)
    # the best parameters that caused the best accuracy
    print "Best Parameters: {}".format(grid.best_params_)
    # the average time it took a model to fit to the data (in seconds)
    print "Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3))
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    print "Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3))


# archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# import the newest csv
credit_card_default = pd.read_csv('../data/credit_card_default.csv')

# 30,000 rows and 24 columns
credit_card_default.shape

# Some descriptive statistics
# We invoke the .T to transpose the matrix for better viewing
credit_card_default.describe().T

# check for missing values, none in this dataset
credit_card_default.isnull().sum()

# Create our feature matrix
X = credit_card_default.drop('default payment next month', axis=1)

# create our response variable
y = credit_card_default['default payment next month']

# get our null accuracy rate
y.value_counts(normalize=True)

#0    0.7788
#1    0.2212
#So, the accuracy to beat, in this case, is 77.88%, which is the percentage of people who did not default (0 meaning false to default).

#### Creating a Baseline Machine Learning Pipeline

# instantiate the four machine learning models
lr = LogisticRegression()
knn = KNeighborsClassifier()
d_tree = DecisionTreeClassifier()
forest = RandomForestClassifier()

get_best_model_and_accuracy(lr, lr_params, X, y)

#Best Accuracy: 0.809566666667
#Best Parameters: {'penalty': 'l1', 'C': 0.1}
#Average Time to Fit(s): 0.602
#Average Time to Score(s): 0.002

get_best_model_and_accuracy(knn, knn_params, X, y)

#Best Accuracy: 0.760233333333
#Best Parameters: {'n_neighbors': 7}
#Average Time to Fit(s): 0.035
#Average Time to Score(s): 0.88

'''
KNN is a distance-based model, in that it uses a metric of closeness in space that assumes that all features are on the same scale, which we already know that our data is not on. So, for KNN, we will have to construct a more complicated pipeline to more accurately assess its baseline performance, using the following code:
'''

# bring in some familiar modules for dealing with this sort of thing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# construct pipeline parameters based on the parameters
# for KNN on its own
knn_pipe_params = {'classifier__{}'.format(
    k): v for k, v in knn_params.iteritems()}

# KNN requires a standard scalar due to using Euclidean distance # as the main equation for predicting observations
knn_pipe = Pipeline([('scale', StandardScaler()), ('classifier', knn)])

# quick to fit, very slow to predict
get_best_model_and_accuracy(knn_pipe, knn_pipe_params, X, y)

print knn_pipe_params  # {'classifier__n_neighbors': [1, 3, 5, 7]}

#Best Accuracy: 0.8008
#Best Parameters: {'classifier__n_neighbors': 7}
#Average Time to Fit(s): 0.035
#Average Time to Score(s): 6.723

get_best_model_and_accuracy(d_tree, tree_params, X, y)

#Best Accuracy: 0.820266666667
#Best Parameters: {'max_depth': 3}
#Average Time to Fit(s): 0.158
#Average Time to Score(s): 0.002

get_best_model_and_accuracy(forest, forest_params, X, y)

#Best Accuracy: 0.819566666667
#Best Parameters: {'n_estimators': 50, 'max_depth': 7}
#Average Time to Fit(s): 1.107
#Average Time to Score(s): 0.044


'''
TYPE OF FEATURES SELECTION 

1- FEATURE SELECTION BASED ON STATISTICS

'''

credit_card_default.corr()

'''
It is worth noting that Pearson’s correlation generally requires that each column be normally distributed (which we are not assuming). We can also largely ignore this requirement because our dataset is large (over 500 is the threshold).
'''
# using seaborn to generate heatmaps
import seaborn as sns
import matplotlib.style as style
# Use a clean stylizatino for our charts and graphs
style.use('fivethirtyeight')

sns.heatmap(credit_card_default.corr())

credit_card_default.corr()['default payment next month']
# These are the features that we might assume are going to be useful. Let's use pandas filtering to isolate features that have at least .2 correlation (positive or negative).
credit_card_default.corr()['default payment next month'].abs() > .2

# store the features
highly_correlated_features = credit_card_default.columns[credit_card_default.corr()[
    'default payment next month'].abs() > .2]

highly_correlated_features

#Index([u'PAY_0', u'PAY_2', u'PAY_3', u'PAY_4', u'PAY_5',
#       u'default payment next month'],
#      dtype='object')


# drop the response variable
highly_correlated_features = highly_correlated_features.drop(
    'default payment next month')

highly_correlated_features


# only include the five highly correlated features
X_subsetted = X[highly_correlated_features]

get_best_model_and_accuracy(d_tree, tree_params, X_subsetted, y)

# barely worse, but about 20x faster to fit the model
#Best Accuracy: 0.819666666667
#Best Parameters: {'max_depth': 3}
#Average Time to Fit(s): 0.01
#Average Time to Score(s): 0.002

'''
Let's bring back our scikit-learn pipelines and include our correlation choosing methodology as a part of our preprocessing phase. To do this, we will have to create a custom transformer that invokes the logic we just went through, as a pipeline-ready class.

We will call our class the CustomCorrelationChooser and it will have to implement both a fit and a transform logic, which are:

The fit logic will select columns from the features matrix that are higher than a specified threshold
The transform logic will subset any future datasets to only include those columns that were deemed important

'''

from sklearn.base import TransformerMixin, BaseEstimator


class CustomCorrelationChooser(TransformerMixin, BaseEstimator):
    def __init__(self, response, cols_to_keep=[], threshold=None):
        # store the response series
        self.response = response
        # store the threshold that we wish to keep
        self.threshold = threshold
        # initialize a variable that will eventually
        # hold the names of the features that we wish to keep
        self.cols_to_keep = cols_to_keep

    def transform(self, X):
        # the transform method simply selects the appropiate
        # columns from the original dataset
        return X[self.cols_to_keep]

    def fit(self, X, *_):
        # create a new dataframe that holds both features and response
        df = pd.concat([X, self.response], axis=1)
        # store names of columns that meet correlation threshold
        self.cols_to_keep = df.columns[df.corr(
        )[df.columns[-1]].abs() > self.threshold]
        # only keep columns in X, for example, will remove response variable
        self.cols_to_keep = [c for c in self.cols_to_keep if c in X.columns]
        return self


# instantiate our new feature selector
ccc = CustomCorrelationChooser(threshold=.2, response=y)
ccc.fit(X)

ccc.cols_to_keep

#['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']


'''
We see that the transform method has eliminated the other columns and kept only the features that met our .2 correlation threshold. Now, let's put it all together in our pipeline, with the help of the following code:
'''

# instantiate our feature selector with the response variable set
ccc = CustomCorrelationChooser(response=y)

# make our new pipeline, including the selector
ccc_pipe = Pipeline([('correlation_select', ccc),
                     ('classifier', d_tree)])

# make a copy of the decisino tree pipeline parameters
ccc_pipe_params = deepcopy(tree_pipe_params)

# update that dictionary with feature selector specific parameters
ccc_pipe_params.update({
    'correlation_select__threshold': [0, .1, .2, .3]})

# {'correlation_select__threshold': [0, 0.1, 0.2, 0.3], 'classifier__max_depth': [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
print ccc_pipe_params

# better than original (by a little, and a bit faster on
# average overall
get_best_model_and_accuracy(ccc_pipe, ccc_pipe_params, X, y)

#Best Accuracy: 0.8206
#Best Parameters: {'correlation_select__threshold': 0.1, 'classifier__max_depth': 5}
#Average Time to Fit(s): 0.105
#Average Time to Score(s): 0.003

# check the threshold of .1
ccc = CustomCorrelationChooser(threshold=0.1, response=y)
ccc.fit(X)

# check which columns were kept
ccc.cols_to_keep
#['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


'''
TYPE OF FEATURES SELECTION 

2- FEATURE SELECTION USING HYPOTHESIS TESTING

'''

# SelectKBest selects features according to the k highest scores of a given scoring function
from sklearn.feature_selection import SelectKBest

# This models a statistical test known as ANOVA
from sklearn.feature_selection import f_classif

# f_classif allows for negative values, not all do
# chi2 is a very common classification criteria but only allows for positive values
# regression has its own statistical tests
k_best = SelectKBest(f_classif)

# Make a new pipeline with SelectKBest
select_k_pipe = Pipeline([('k_best', k_best),
                          ('classifier', d_tree)])

select_k_best_pipe_params = deepcopy(tree_pipe_params)
# the 'all' literally does nothing to subset
select_k_best_pipe_params.update({'k_best__k': range(1, 23) + ['all']})

# {'k_best__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'all'], 'classifier__max_depth': [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
print select_k_best_pipe_params

# comparable to our results with correlationchooser
get_best_model_and_accuracy(select_k_pipe, select_k_best_pipe_params, X, y)


#Best Accuracy: 0.8206
#Best Parameters: {'k_best__k': 7, 'classifier__max_depth': 5}
#Average Time to Fit(s): 0.102
#Average Time to Score(s): 0.002

k_best = SelectKBest(f_classif, k=7)

# lowest 7 p values match what our custom correlationchooser chose before
# ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

p_values.head(7)

'''
TYPE OF FEATURES SELECTION 

3- MODEL BASED FEATURE SELECTION

'''

# create a brand new decision tree classifier
tree = DecisionTreeClassifier()

tree.fit(X, y)

# note that we have some other features in play besides what our last two selectors decided for us

importances = pd.DataFrame({'importance': tree.feature_importances_,
                            'feature': X.columns}).sort_values('importance', ascending=False)

importances.head()

# similar to SelectKBest, but not with statistical tests
from sklearn.feature_selection import SelectFromModel

# instantiate a class that choses features based
# on feature importances according to the fitting phase
# of a separate decision tree classifier
select_from_model = SelectFromModel(DecisionTreeClassifier(),
                                    threshold=.05)
selected_X = select_from_model.fit_transform(X, y)
selected_X.shape

#(30000, 9)


# to speed things up a bit in the future
tree_pipe_params = {'classifier__max_depth': [1, 3, 5, 7]}

from sklearn.pipeline import Pipeline

# create a SelectFromModel that is tuned by a DecisionTreeClassifier
select = SelectFromModel(DecisionTreeClassifier())

select_from_pipe = Pipeline([('select', select),
                             ('classifier', d_tree)])

select_from_pipe_params = deepcopy(tree_pipe_params)

select_from_pipe_params.update({
    'select__threshold': [.01, .05, .1, .2, .25, .3, .4, .5, .6, "mean", "median", "2.*mean"],
    'select__estimator__max_depth': [None, 1, 3, 5, 7]
})

# {'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 'select__estimator__max_depth': [None, 1, 3, 5, 7], 'classifier__max_depth': [1, 3, 5, 7]}
print select_from_pipe_params


get_best_model_and_accuracy(select_from_pipe,
                            select_from_pipe_params,
                            X, y)

# not better than original
#Best Accuracy: 0.820266666667
#Best Parameters: {'select__threshold': 0.01, 'select__estimator__max_depth': None, 'classifier__max_depth': 3}
#Average Time to Fit(s): 0.192
#Average Time to Score(s): 0.002

# set the optimal params to the pipeline
select_from_pipe.set_params(**{'select__threshold': 0.01,
                               'select__estimator__max_depth': None,
                               'classifier__max_depth': 3})

# fit our pipeline to our data
select_from_pipe.steps[0][1].fit(X, y)

# list the columns that the SVC selected by calling the get_support() method from SelectFromModel
X.columns[select_from_pipe.steps[0][1].get_support()]


#[u'LIMIT_BAL', u'SEX', u'EDUCATION', u'MARRIAGE', u'AGE', u'PAY_0', u'PAY_2', u'PAY_3', u'PAY_6', u'BILL_AMT1', u'BILL_AMT2',
#    u'BILL_AMT3', u'BILL_AMT4', u'BILL_AMT5', u'BILL_AMT6', u'PAY_AMT1', u'PAY_AMT2', u'PAY_AMT3', u'PAY_AMT4', u'PAY_AMT5', u'PAY_AMT6']


'''
TYPE OF FEATURES SELECTION 

4- LINEAR MODEL COEFFICIENTS AS FEATURE SELECTION METRIC

'''

# a new selector that uses the coefficients from a regularized logistic regression as feature importances
logistic_selector = SelectFromModel(LogisticRegression())

# make a new pipeline that uses coefficients from LogistisRegression as a feature ranker
regularization_pipe = Pipeline([('select', logistic_selector),
                                ('classifier', tree)])

regularization_pipe_params = deepcopy(tree_pipe_params)

# try l1 regularization and l2 regularization
regularization_pipe_params.update({
    'select__threshold': [.01, .05, .1, "mean", "median", "2.*mean"],
    'select__estimator__penalty': ['l1', 'l2'],
})

# {'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 'classifier__max_depth': [1, 3, 5, 7], 'select__estimator__penalty': ['l1', 'l2']}
print regularization_pipe_params


get_best_model_and_accuracy(regularization_pipe,
                            regularization_pipe_params,
                            X, y)


# better than original, in fact the best so far, and much faster on the scoring side
#Best Accuracy: 0.821166666667 Best Parameters: {'select__threshold': 0.01, 'classifier__max_depth': 5, 'select__estimator__penalty': 'l1'}
#Average Time to Fit(s): 0.51
#Average Time to Score(s): 0.001


# set the optimal params to the pipeline
regularization_pipe.set_params(**{'select__threshold': 0.01,
                                  'classifier__max_depth': 5,
                                  'select__estimator__penalty': 'l1'})

# fit our pipeline to our data
regularization_pipe.steps[0][1].fit(X, y)

# list the columns that the Logisti Regression selected by calling the get_support() method from SelectFromModel
X.columns[regularization_pipe.steps[0][1].get_support()]

#[u'SEX', u'EDUCATION', u'MARRIAGE', u'PAY_0',
#    u'PAY_2', u'PAY_3', u'PAY_4', u'PAY_5']

# SVC is a linear model that uses linear supports to
# seperate classes in euclidean space
# This model can only work for binary classification tasks
from sklearn.svm import LinearSVC

# Using a support vector classifier to get coefficients
svc_selector = SelectFromModel(LinearSVC())

svc_pipe = Pipeline([('select', svc_selector),
                     ('classifier', tree)])

svc_pipe_params = deepcopy(tree_pipe_params)

svc_pipe_params.update({
    'select__threshold': [.01, .05, .1, "mean", "median", "2.*mean"],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

# 'select__estimator__loss': ['squared_hinge', 'hinge'], 'select__threshold': [0.01, 0.05, 0.1, 'mean', 'median', '2.*mean'], 'select__estimator__penalty': ['l1', 'l2'], 'classifier__max_depth': [1, 3, 5, 7], 'select__estimator__dual': [True, False]}
print svc_pipe_params

get_best_model_and_accuracy(svc_pipe,
                            svc_pipe_params,
                            X, y)


# better than original, in fact the best so far, and much faster on the scoring side
#Best Accuracy: 0.821233333333
#Best Parameters: {'select__estimator__loss': 'squared_hinge', 'select__threshold': 0.01, 'select__estimator__penalty': 'l1', 'classifier__max_depth': 5, 'select__estimator__dual': False}
#Average Time to Fit(s): 0.989
#Average Time to Score(s): 0.001


# set the optimal params to the pipeline
svc_pipe.set_params(**{'select__estimator__loss': 'squared_hinge',
                       'select__threshold': 0.01,
                       'select__estimator__penalty': 'l1',
                       'classifier__max_depth': 5,
                       'select__estimator__dual': False})

# fit our pipeline to our data
svc_pipe.steps[0][1].fit(X, y)

# list the columns that the SVC selected by calling the get_support() method from SelectFromModel
X.columns[svc_pipe.steps[0][1].get_support()]

#[u'SEX', u'EDUCATION', u'MARRIAGE', u'PAY_0', u'PAY_2', u'PAY_3', u'PAY_5']


'''

At this point, you may be feeling a bit overwhelmed with the information in this chapter. We have presented several ways of performing feature selection, some based on pure statistics and others based on the output of secondary machine learning models. It is natural to wonder how to decide which feature selection method is right for your data. In theory, if you are able to try multiple options, as we did in this chapter, that would be ideal, but we understand that it might not be feasible to do so. The following are some rules of thumbs that you can follow when you are trying to prioritize which feature selection module is more likely to offer greater results:

If your features are mostly categorical, you should start by trying to implement a SelectKBest with a Chi2 ranker or a tree-based model selector.
If your features are largely quantitative (like ours were), using linear models as model-based selectors and relying on correlations tends to yield greater results, as was shown in this chapter.
If you are solving a binary classification problem, using a Support Vector Classification model along with a SelectFromModel selector will probably fit nicely, as the SVC tries to find coefficients to optimize for binary classification tasks.
A little bit of EDA can go a long way in manual feature selection. The importance of having domain knowledge in the domain from which the data originated cannot be understated.
That being said, these are meant only to be used as guidelines. As a data scientist, ultimately you decide which features you wish to keep to optimize the metric of your choosing. The methods that we provide in this text are here to help you in your discovery of the latent power of features hidden by noise and multicollinearity.

'''
