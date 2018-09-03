%matplotlib inline
# Load libraries
import numpy as np
from numpy import arange
from numpy import set_printoptions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RandomizedLasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from string import letters
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score, roc_auc_score
from collections import defaultdict
import pprint
from minepy import MINE

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost

sns.set(style="white")
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# Load the data set into a dataframe
df = pd.read_csv('../data/data_2018/NoScore.csv')
# Print dataset
# df.head()

# Print data types for attributes
pd.set_option('display.max_rows', 600)
df.dtypes

# Drop the columns we don't need for our predictor
df = df.drop(['Unnamed: 0', 'pupID', 'chal.time',
              'litterID', 'dob.pups', 'removed'],  1)

# print list(df)
#['outcome',
# 'age.challenge',
# 'sex',
# 'chal.status',
# 'weight.challenge',
# 'v18.hour.post.challenge',
# 'v18.weight',
# 'v18.energy.high',
# 'v18.energy.low',
# 'v18.righting.response.high',
# 'v18.righting.response.low',
# 'v24.hour.post.challenge',
# 'v24.weight',
# 'v24.energy.high',
# 'v24.energy.low',
# 'v24.righting.response.high',
# 'v24.righting.response.low',
# 'weight.change.0.24',
# 'weight.change.0.18',
# 'weight.change.18.24',
# 'direction.of.weight.change.18.24',
# 'change.righting.high',
# 'change.righting.low']

# print df.head()


columns_to_encode = ['sex', 'chal.status', 'v18.energy.high', 'v18.energy.low', 'v18.righting.response.high', 'v18.righting.response.low', 'v24.energy.high',
                     'v24.energy.low', 'v24.righting.response.high', 'v24.righting.response.low', 'direction.of.weight.change.18.24', 'change.righting.high', 'change.righting.low']
df_encoded = pd.get_dummies(df, drop_first=True, columns=columns_to_encode)

pd.set_option('precision', 3)
df_encoded.describe()

# Examine the sape of the data now
# print df.shape
# we have 222 rows and 23 Columns (22 features and 1 outcome)

# Create our feature matrix
X = df_encoded.drop('outcome', axis=1)

# create our response variable
y = df_encoded['outcome']

# get our null accuracy rate
y.value_counts(normalize=True)

# die     0.505
# live    0.495
# The accuracy to beat is 50.5 %
# Name:
#    outcome, dtype:
#        float64


def preprocess_data(df):
    df = df.drop(['Unnamed: 0', 'pupID', 'chal.time',
                  'litterID', 'dob.pups', 'removed'],  1)
    columns_to_encode = ['sex', 'chal.status', 'v18.energy.high', 'v18.energy.low', 'v18.righting.response.high', 'v18.righting.response.low', 'v24.energy.high',
                         'v24.energy.low', 'v24.righting.response.high', 'v24.righting.response.low', 'direction.of.weight.change.18.24', 'change.righting.high', 'change.righting.low']
    df_encoded = pd.get_dummies(df, drop_first=True, columns=columns_to_encode)
    df_encoded['outcome'] = pd.factorize(df_encoded['outcome'])[0]
    # Create our feature matrix
    X = df_encoded.drop('outcome', axis=1)
    # create our response variable
    y = df_encoded['outcome']
    return X, y

# Creating a Baseline Machine Learning Pipeline

def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,  # the model to grid search
                        params,  # the parameter set to try
                        error_score=0.)  # if a parameter set raises an error, continue and set the performance as a big, fat 0
    grid.fit(X, y)  # fit the model and parameters
    # our classical metric for performance
    print "Best Accuracy: {}".format(grid.best_score_)
    # the best parameters that caused the best accuracy
    print "Best Parameters: {}".format(grid.best_params_)
    # the average time it took a model to fit to the data (in seconds)
    print "Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3))
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in
    # real-time analysis
    print "Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3))

# instantiate the four machine learning models
lr = LogisticRegression()
knn = KNeighborsClassifier()
d_tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
gb = GradientBoostingClassifier()
xg = xgboost.XGBClassifier()
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
forest_params = {'n_estimators': [10, 50, 100, 500, 1000], 
                'max_depth': [None, 1, 3, 5, 7]}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': [10, 50, 100, 500, 1000],
    #'max_features': 0.2,
    'max_depth': [1, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

xg_params = {
    'n_estimators' : [10, 50, 100, 500, 1000],
    # Parameters that we are going to tune.
    'max_depth': [1, 3, 5, 7],
    'min_child_weight': [1,2,3,4,5,6,7],
    'eta': [0.3, 0.2, 0.4, 0.5],
    'subsample': [1,2,3,4],
    'colsample_bytree': [1,2,3,4],
    # Other parameters
    'objective': ['reg:linear','binary:logistic'],
    'eval_metric' : ['auc', 'error'],
}
#-------------------------------------------------------------#


get_best_model_and_accuracy(lr, lr_params, X, y)  # Logsitc Regression
get_best_model_and_accuracy(knn, knn_params, X, y)  # KNN

'''
KNN is a distance-based model, in that it uses a metric of closeness in space that assumes that all features are on the same scale, which we already know that our data is not on. So, for KNN, we will have to construct a more complicated pipeline to more accurately assess its baseline performance, using the following code:
'''

# bring in some familiar modules for dealing with this sort of thing
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

# construct pipeline parameters based on the parameters
# for KNN on its own
knn_pipe_params = {'classifier__{}'.format(
    k): v for k, v in knn_params.iteritems()}

# KNN requires a standard scalar due to using Euclidean distance # as the
# main equation for predicting observations
knn_pipe = Pipeline([('scale', StandardScaler()), ('classifier', knn)])

# quick to fit, very slow to predict
get_best_model_and_accuracy(knn_pipe, knn_pipe_params, X, y)


# Best Accuracy: 0.8008
# Best Parameters: {'classifier__n_neighbors': 7}
# Average Time to Fit(s): 0.035
# Average Time to Score(s): 6.723

get_best_model_and_accuracy(d_tree, tree_params, X, y)

# Best Accuracy: 0.820266666667
# Best Parameters: {'max_depth': 3}
# Average Time to Fit(s): 0.158
# Average Time to Score(s): 0.002

get_best_model_and_accuracy(forest, forest_params, X, y)

# Best Accuracy: 0.819566666667
# Best Parameters: {'n_estimators': 50, 'max_depth': 7}
# Average Time to Fit(s): 1.107
# Average Time to Score(s): 0.044

'''
TYPE OF FEATURES SELECTION 

1- FEATURE SELECTION BASED ON STATISTICS

'''

df_encoded.corr()

'''
It is worth noting that Pearson’s correlation generally requires that each column be normally distributed (which we are not assuming). We can also largely ignore this requirement because our dataset is large (over 500 is the threshold).
'''
# using seaborn to generate heatmaps
import seaborn as sns
import matplotlib.style as style
# Use a clean stylizatino for our charts and graphs
style.use('fivethirtyeight')

sns.heatmap(df_encoded.corr())

df_encoded.corr()['outcome']
# These are the features that we might assume are going to be useful.
# Let's use pandas filtering to isolate features that have at least .2
# correlation (positive or negative).
df_encoded.corr()['outcome'].abs() > .2

# store the features
highly_correlated_features = df_encoded.columns[df_encoded.corr()['outcome'].abs() > .2]
#highly_correlated_features
# drop the response variable
highly_correlated_features = highly_correlated_features.drop('outcome')
#highly_correlated_features
# only include the five highly correlated features
X_subsetted = X[highly_correlated_features]
get_best_model_and_accuracy(d_tree, tree_params, X_subsetted, y)

# Doing it more eleganlty


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
tree_pipe_params = {'classifier__n_estimators':[10, 50, 100, 500, 1000],
'classifier__max_depth': [1, 3, 5, 7],
'classifier__min_child_weight':[1, 2, 3, 4, 5, 6, 7],
'classifier__subsample':[1, 2, 3, 4],
'classifier__colsample_bytree':[1, 2, 3, 4],
'classifier__objective': ['reg:linear', 'binary:logistic']}

ccc = CustomCorrelationChooser(response=y)

# make our new pipeline, including the selector
ccc_pipe = Pipeline([('correlation_select', ccc),
                     ('classifier', xg)])

# make a copy of the decisino tree pipeline parameters
ccc_pipe_params = deepcopy(tree_pipe_params)

# update that dictionary with feature selector specific parameters
ccc_pipe_params.update({
    'correlation_select__threshold': [0, .1, .2, .3]})

# average overall
get_best_model_and_accuracy(ccc_pipe, ccc_pipe_params, X, y)



# check the threshold of .1
ccc = CustomCorrelationChooser(threshold=0.1, response=y)
ccc.fit(X)
# check which columns were kept
ccc.cols_to_keep
#['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'


'''
TYPE OF FEATURES SELECTION 

2- FEATURE SELECTION USING HYPOTHESIS TESTING

'''

# SelectKBest selects features according to the k highest scores of a
# given scoring function
from sklearn.feature_selection import SelectKBest

# This models a statistical test known as ANOVA
from sklearn.feature_selection import f_classif

# f_classif allows for negative values, not all do
# chi2 is a very common classification criteria but only allows for positive values
# regression has its own statistical tests
k_best = SelectKBest(f_classif)

# Make a new pipeline with SelectKBest
select_k_pipe = Pipeline([('k_best', k_best),
                          ('classifier', xg)])

select_k_best_pipe_params = deepcopy(tree_pipe_params)
# the 'all' literally does nothing to subset
select_k_best_pipe_params.update({'k_best__k': range(1, 20) + ['all']})

# {'k_best__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'all'], 'classifier__max_depth': [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
#print select_k_best_pipe_params

# comparable to our results with correlationchooser
get_best_model_and_accuracy(select_k_pipe, select_k_best_pipe_params, X, y)


#Best Accuracy: 0.8206
#Best Parameters: {'k_best__k': 7, 'classifier__max_depth': 5}
#Average Time to Fit(s): 0.102
#Average Time to Score(s): 0.002

k_best = SelectKBest(f_classif, k=7)
k_best.fit_transform(X, y)
# lowest 7 p values match what our custom correlationchooser chose before
# ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
p_values = pd.DataFrame(
    {'column': X.columns, 'p_value': k_best.pvalues_}).sort_values('p_value')

p_values.head(7)


'''
TYPE OF FEATURES SELECTION 

3- MODEL BASED FEATURE SELECTION

'''

## create a brand new decision tree classifier
#tree = xgboost.XGBClassifier()
#
#tree.fit(X, y)
#
## note that we have some other features in play besides what our last two
## selectors decided for us
#
#importances = pd.DataFrame({'importance': tree.feature_importances_,
#                            'feature': X.columns}).sort_values('importance', ascending=False)
#
#importances.head()
#
## similar to SelectKBest, but not with statistical tests
#from sklearn.feature_selection import SelectFromModel
#
## instantiate a class that choses features based
## on feature importances according to the fitting phase
## of a separate decision tree classifier
#select_from_model = SelectFromModel(DecisionTreeClassifier(),
#                                    threshold=.05)
#selected_X = select_from_model.fit_transform(X, y)
#selected_X.shape
#
##(30000, 9)


# to speed things up a bit in the future
# instantiate our feature selector with the response variable set
tree_pipe_params = {'classifier__n_estimators': [10, 50, 100, 500, 1000],
                    'classifier__max_depth': [1, 3, 5, 7],
                    'classifier__min_child_weight': [1, 2, 3, 4, 5, 6, 7],
                    'classifier__subsample': [1, 2, 3, 4],
                    'classifier__colsample_bytree': [1, 2, 3, 4],
                    'classifier__objective': ['reg:linear', 'binary:logistic']}
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# create a SelectFromModel that is tuned by a DecisionTreeClassifier
select = SelectFromModel(xgboost.XGBClassifier())

select_from_pipe = Pipeline([('select', select),
                             ('classifier', xg)])

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


'''
TYPE OF FEATURES SELECTION 

4- LINEAR MODEL COEFFICIENTS AS FEATURE SELECTION METRIC

'''

# a new selector that uses the coefficients from a regularized logistic
# regression as feature importances
logistic_selector = SelectFromModel(LogisticRegression())

# make a new pipeline that uses coefficients from LogistisRegression as a
# feature ranker
regularization_pipe = Pipeline([('select', logistic_selector),
                                ('classifier', xg)])

tree_pipe_params = {'classifier__colsample_bytree': 1, 'classifier__subsample': 0.5, 'classifier__n_estimators': 100,
    'classifier__min_child_weight': 3, 'classifier__max_depth': 1, 'classifier__objective': 'binary:logistic'}

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

# set the optimal params to the pipeline
regularization_pipe.set_params(**{'select__threshold': "median",
                                  'classifier__max_depth': 1,
                                  'select__estimator__penalty': 'l1'})

# fit our pipeline to our data
regularization_pipe.steps[0][1].fit(X, y)

# list the columns that the Logisti Regression selected by calling the
# get_support() method from SelectFromModel
X.columns[regularization_pipe.steps[0][1].get_support()]


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


{'classifier__colsample_bytree': 1, 'classifier__subsample': 0.5, 'select__estimator__penalty': 'l2', 'select__estimator__loss': 'squared_hinge', 'classifier__n_estimators': 50, 'select__estimator__dual': True, 'classifier__min_child_weight': 3, 'select__threshold': 0.1, 'classifier__max_depth': 5, 'classifier__objective': 'binary:logistic'}
