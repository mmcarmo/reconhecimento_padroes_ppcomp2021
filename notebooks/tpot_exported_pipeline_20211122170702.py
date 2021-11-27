import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8495825748540581
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_union(
            RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.15000000000000002, n_estimators=100), step=0.45),
            FunctionTransformer(copy)
        )
    ),
    LGBMClassifier(boosting_type="gbdt", class_weight=None, max_depth=5, min_child_samples=17, n_estimators=100, n_jobs=1, silent=True, subsample=0.9500000000000002)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
