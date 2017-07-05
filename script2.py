import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV

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
pca2_results_train = pca.fit_transform(train.drop(["y"].drop["ID"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"].drop["ID"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-2]
    test['pca_' + str(i)] = pca2_results_test[:, i-2]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-2]
    test['ica_' + str(i)] = ica2_results_test[:, i-2]
    

y_train = train["y"]
y_mean = np.mean(y_train)

y = np.array(train1["y"])
X=np.array(train1.drop('y', axis=1))
import xgboost as xgb

# prepare dict of params for xgboost to run wi
# form DMatrices for Xgboost training
x_train=train.drop('y', axis=1)
dtrain = xgb.DMatrix(x_train, y_train)
test1 = test
dtest = xgb.DMatrix(test)

print 'Split sucessful'

# xgboost, cross-validation

xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

cv_result = xgb.cv(xgb_params, 
                   dtrain,
		               nfold=5, 
                   num_boost_round=5000,
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
		               seed=0
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
clf = BaggingRegressor(n_estimators=25)
clf.fit(X,y)
y_pred1 = clf.predict(X)



# train model
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
y_pred2 = model.predict(dtrain)
y_pred3 = np.append(y_pred1, y_pred2, axis=1)


clf2 = BaggingRegressor(n_estimators=25)
clf2.fit(y_pred3,y)
# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

y_pred = model.predict(dtest)




output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica-master.csv'.format(xgb_params['max_depth']), index=False)
