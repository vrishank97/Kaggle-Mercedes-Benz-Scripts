import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import cross_val_score, GridSearchCV
from lightgbm import LGBMRegressor

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#train=train[train.y<250]

train['Z0'] = train.groupby('X0')['y'].transform('median')
train['Z1'] = train.groupby('X1')['y'].transform('median')
train['Z2'] = train.groupby('X2')['y'].transform('median')
train['Z3'] = train.groupby('X3')['y'].transform('median')
train['Z4'] = train.groupby('X4')['y'].transform('median')
train['Z5'] = train.groupby('X5')['y'].transform('median')
train['Z6'] = train.groupby('X6')['y'].transform('median')
train['Z8'] = train.groupby('X8')['y'].transform('median')

test['Z0']=test['X0'].map(train.groupby('X0')['y'].median())
test['Z1']=test['X1'].map(train.groupby('X1')['y'].median())
test['Z2']=test['X2'].map(train.groupby('X2')['y'].median())
test['Z3']=test['X3'].map(train.groupby('X3')['y'].median())
test['Z4']=test['X4'].map(train.groupby('X4')['y'].median())
test['Z5']=test['X5'].map(train.groupby('X5')['y'].median())
test['Z6']=test['X6'].map(train.groupby('X6')['y'].median())
test['Z8']=test['X8'].map(train.groupby('X8')['y'].median())

train = train.sort_values(by='ID')

med = train['y'].median()
#print(med)

test['Z0']=test['Z0'].fillna(med)
test['Z1']=test['Z1'].fillna(med)
test['Z2']=test['Z2'].fillna(med)
test['Z3']=test['Z3'].fillna(med)
test['Z4']=test['Z4'].fillna(med)
test['Z5']=test['Z5'].fillna(med)
test['Z6']=test['Z6'].fillna(med)
test['Z8']=test['Z8'].fillna(med)

train=train.drop('X0', axis=1)
train=train.drop('X1', axis=1)
train=train.drop('X2', axis=1)
train=train.drop('X3', axis=1)
train=train.drop('X4', axis=1)
train=train.drop('X5', axis=1)
train=train.drop('X6', axis=1)
train=train.drop('X8', axis=1)


test=test.drop('X0', axis=1)
test=test.drop('X1', axis=1)
test=test.drop('X2', axis=1)
test=test.drop('X3', axis=1)
test=test.drop('X4', axis=1)
test=test.drop('X5', axis=1)
test=test.drop('X6', axis=1)
test=test.drop('X8', axis=1)


# process columns, apply LabelEncoder to categorical features
'''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
'''
# shape        
#print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]
    
    train['grp_' + str(i)] = grp_results_train[:,i-1]
    test['grp_' + str(i)] = grp_results_test[:, i-1]
    
    train['srp_' + str(i)] = srp_results_train[:,i-1]
    test['srp_' + str(i)] = srp_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)

y = np.array(train["y"])
X=np.array(train.drop('y', axis=1))

model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=675, max_bin=25, subsample_for_bin=50000, objective=None, min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True)
lol= cross_val_score(model, X, y, cv=5)
print (lol)
print (np.mean(lol), np.std(lol))

model.fit(X, y)
y0=model.predict(test)
test['y']=y0
output=test[['ID', 'y']]
output.to_csv('output.csv', index=False)




#print (lol)