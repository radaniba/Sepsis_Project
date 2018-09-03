import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.873039215686
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=5.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.0001)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=17, min_samples_split=18)),
    BernoulliNB(alpha=0.001, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
