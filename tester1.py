import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

from sklearn.decomposition import PCA, FastICA
n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-2]
    test['pca_' + str(i)] = pca2_results_test[:, i-2]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-2]
    test['ica_' + str(i)] = ica2_results_test[:, i-2]
    
y = train["y"]
y_mean = np.mean(y)
X=train.drop('y', axis=1)

print 'Split sucessful'


kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = RandomForestClassifier(n_estimators=10)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = GradientBoostingRegressor(n_estimators=100)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = BaggingRegressor(n_estimators=10)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = RandomForestClassifier(n_estimators=10)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = AdaBoostRegressor(n_estimators=10)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf = ExtraTreesRegressor(n_estimators=10)
  clf = clf.fit(X_train, y_train)
  print (clf.score(X_test, y_test))

'''

# train model
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)


y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica-master.csv'.format(xgb_params['max_depth']), index=False)
'''