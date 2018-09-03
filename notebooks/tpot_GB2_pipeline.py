import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.933088235294
exported_pipeline = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.8, n_estimators=100), step=0.55),
    MaxAbsScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.5, min_samples_leaf=10, min_samples_split=16, n_estimators=100, subsample=0.85)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
