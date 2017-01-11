# Univariate feature selection on Pima Indians diabetes data set
# Classification Problem
# Data set info: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
# Features are:
# 0. Number of times pregnant 
# 1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 2. Diastolic blood pressure (mm Hg)
# 3. Triceps skin fold thickness (mm)
# 4. 2-Hour serum insulin (mu U/ml)
# 5. Body mass index (weight in kg/(height in m)^2)
# 6. Diabetes pedigree function
# 7. Age (years)
# Output(y) 8. Class variable (0 or 1)
#
# Tested algorithm has 76% sensitivity and specificity


import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Making sure everything is reproducible
np.random.seed(0)

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
col_names = ['preg', 'plas', 'pres', 'skin', 'insulin', 'mass', 'pedi', 'age', 'class']
data_all = pd.read_csv(url, names=col_names)
# print "Names: \n", list(data_all.columns.values)
x = data_all.values[:, 0:8]
y = data_all.values[:, 8]


# Feature selection using chi2 with KBest
features = SelectKBest(score_func=chi2, k=4)
fit = features.fit(x, y)
np.set_printoptions(precision=4)
print "Feature ranking by Chi2 Test: "
print sorted(zip(map(lambda x: round(x, 4), fit.scores_), col_names), reverse=True)


### Pearson Correlation & p-value.
# Measures "only" the linear relation between feature & response.
# P-value: smaller the value, more confidence the probability that the feature is important
from scipy.stats import pearsonr
corr_pvalue = []
print "\n"
for i in range(x.shape[1]):
    corr_pvalue.append(pearsonr(x[:, i], y))
    np.set_printoptions(precision=3)
    print "Correlation, P-Value for feature {}: {}".format(col_names[i], corr_pvalue[i])

pearson_list = [i[0] for i in corr_pvalue]
print "Pearson Correlation (linear) of All Features: "
print sorted(zip(map(lambda x: round(x, 4), pearson_list), col_names), reverse=True)


### Sklearn's F-value and p-value.
# F-value: Higher the more important the feature is.
from sklearn.feature_selection import f_regression
f_p = f_regression(x[:, :], y)
print "\nF-value: ", f_p[0], "\nP-value: ", f_p[1]
print "Feature Ranking by F-values: "
print sorted(zip(map(lambda x: round(x, 4), f_p[0]), col_names), reverse=True)


### Maximal information coefficient
# Measures strength of linear & non-linear association btwn feature & response
from minepy import MINE
m = MINE()
mic_score = []
print "\n"
for i in range(x.shape[1]):
    m.compute_score(x[:, i], y)
    mic_score.append(m.mic())
    # print "MIC score {}: {}".format(col_names[i], m.mic())
print "MIC Feature Ranking: "
print sorted(zip(map(lambda x: round(x, 4), mic_score), col_names), reverse=True)


# Customized distance correlation calculation.
# This tells us if there is really "NO" relation btwn feature & response
# when the distance correlation is 0
def dist(x, y):
    #1d only
    return np.abs(x[:, None] - y)
    

def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn

def dcov_all(x, y):
    dnx = d_n(x)
    dny = d_n(y)
    
    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = np.sqrt(dc) / (np.sqrt(dvx) * np.sqrt(dvy))
    return dr

print "\n"
distance = []
for i in range(x.shape[1]):
    distance.append(dcov_all(x[:, i], y))
    # print "Distance correlation {}: {}".format(col_names[i], dcov_all(x[:, i], y))

print "Distance Correlation Feature Ranking: "
print sorted(zip(map(lambda x: round(x, 4), distance), col_names), reverse=True)



### Model based ranking with randomforest using CV score
# This measures the non-linearity between feature & response
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=4)

mean_score = []
print "\n"
for i in range(x.shape[1]):
    score = cross_val_score(rf, x[:, i:i+1], y, scoring="r2")
    mean_score.append(round(np.mean(score), 3))
    # print "RF CV score {}: {} | mean: {}".format(col_names[i], score, mean_score)

print "Random Forests Feature Ranking: "
print sorted(zip(map(lambda x: round(x, 4), mean_score), col_names), reverse=True)



