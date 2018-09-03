import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.86887254902
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.95, min_samples_leaf=17, min_samples_split=11, n_estimators=100, subsample=0.85)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
