import numpy as np
import pandas as pd


np.random.seed(0)

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
col_names = ['preg', 'plas', 'pres', 'skin', 'insulin', 'mass', 'pedi', 'age', 'class']
data_all = pd.read_csv(url, names=col_names)

x = data_all.values[:, 0:8]
y = data_all.values[:, 8]


# Stability selection using RandomizedLasso
from sklearn.linear_model import RandomizedLasso

rlasso = RandomizedLasso(alpha=0.0025)
rlasso.fit(x, y)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 col_names), reverse=True)


# Recursive Feature Elimination. 
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


estimator = SVC(kernel="linear")
#rank all features, i.e continue the elimination until the last one
rfe = RFE(estimator, step=1)
rfe.fit(x, y)
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), col_names))

