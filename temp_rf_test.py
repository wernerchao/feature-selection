import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

np.random.seed(42)

#Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
col_names = ['preg', 'plas', 'pres', 'skin', 'insulin', 'mass', 'pedi', 'age', 'class']
data_all = pd.read_csv(url, names=col_names)

x = data_all.values[:, 0:8]
y = data_all.values[:, 8]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

rf = RandomForestClassifier(n_estimators=50, max_depth=4)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
# rf_score = rf.score(x_test, y_test)
rf_recall = recall_score(y_test, rf_pred, average='binary')
rf_precision = precision_score(y_test, rf_pred, average='binary')
print "Random Forest Recall Score: ", rf_recall
print "Random Forest Precision Score: ", rf_precision

svc = SVC(kernel='poly')
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_recall = recall_score(y_test, svc_pred, average='binary')
svc_precision = precision_score(y_test, svc_pred, average='binary')
print "SVC Recall Score: ", svc_recall
print "SVC Precision Score: ", svc_precision



# Take out features: insulin, skin, TODO[pressure]
temp = data_all.drop(['insulin', 'skin', 'class'], axis=1)

x_new = temp.values
x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_new, y, test_size=0.15, random_state=42)

rf_new = RandomForestClassifier(n_estimators=50, max_depth=4)
rf_new.fit(x_new_train, y_new_train)
rf_new_pred = rf_new.predict(x_new_test)
# rf_new_score = rf_new.score(x_new_test, y_new_test)
rf_new_recall = recall_score(y_test, rf_new_pred, average='binary')
rf_new_precision = precision_score(y_test, rf_new_pred, average='binary')
print "NEW Random Forest Recall Score: ", rf_new_recall
print "NEW Random Forest Precision Score: ", rf_new_precision


# svc_new = SVC()
# svc_new.fit(x_new_train, y_new_train)
# svc_new_pred = svc_new.predict(x_new_test)
# svc_score = recall_score(y_test, svc_pred, average='binary')
# print "SVC Score: ", svc_new_pred