# Univariate feature selection on Pima Indians diabetes data set
# Data set info: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
col_names = ['preg', 'plas', 'pres', 'skin', 'insulin', 'mass', 'pedi', 'age', 'class']
data_all = pd.read_csv(url, names=col_names)
x = data_all.values[:, 0:8]
y = data_all.values[:, 8]

# Feature selection using chi2 with KBest
features = SelectKBest(score_func=chi2, k=4)
fit = features.fit(x, y)
np.set_printoptions(precision=3)
print "\nScore for each column: \n", fit.scores_
print x[0]

new_x = fit.transform(x)
print "\nSelected features are: \n", new_x[0:3]
