import pandas as pd
import numpy as np
import sklearn
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

def write_hyperparams(hyperparams, fileName):
	f = open(fileName, 'w')
	f.write(' '.join(str(x) for x in hyperparams) + '\n')
	f.close()

def read_hyperparams(fileName):
	f = open(fileName, 'r')
	n_folds, n_estimators, max_depth, max_features, \
		score = f.read().split()
	return int(n_folds), int(n_estimators), int(max_depth), \
		max_features, float(score)

# define some constants
xgboost_data_dir = '../xgboost_data/'
data_dir = '../data/'
hyperParamFile = 'hyperparams_cat.txt'
n_folds = 3
verbose_tree = 3
verbose_grid = 0
n_jobs = 3
random_state = 1
criterion = 'mae'

# For xgboost
param = {}
param['objective'] = 'reg:gamma'
param['nthread'] = n_jobs
# how to evaluate the tree
param['eval_metric'] = 'mae'
# maximum depth of the tree (3-10)
param['max_depth'] = 6
# eta is the learning rate (0.01-0.02)
param['eta'] = 0.05
# prints messages to screen (1 silences these)
param['silent'] = 1
param['tree_method'] = 'auto'
# L2 regularization term
param['lambda'] = 0
# L1 regularization term
param['alpha'] = 0
# number of training rounds
num_boost_rounds = 500

# read in the train dataset
print('loading training data...')
# for xgboost
dtrain = xgb.DMatrix(xgboost_data_dir+'continuous_selected.dat')

evaluations = xgb.cv(params=param, dtrain=dtrain, num_boost_round=num_boost_rounds,
			nfold=n_folds)
print(evaluations)

tree = xgb.train( param, dtrain, num_boost_rounds)
tree.save_model('prac.model')

dtest = xgb.DMatrix(xgboost_data_dir+'continuous_test_selected.dat')
y_pred = tree.predict(dtest)

df_pred_cat = pd.DataFrame(y_pred)
df_pred_cat.to_csv(path_or_buf=xgboost_data_dir+'allstateClaims_cont.csv')
print('predictions written to allstateClaims_cont.csv')

print(y_pred)

import sys
sys.exit('finished training xgboost')

# for sklearn
df_cont = pd.read_csv('continuous_selected.csv', header=0, index_col=0)
df_y = pd.read_csv('y_all.csv', header=0, index_col=0)

X = df_cont.values
y = np.ravel(df_y.values)

# create a Random Forest Classifier
print('creating a model...')
n_estimators_def = 100
max_features_def = 'sqrt'
max_depth_def = 10
#n_estimators_range = [10, 50]
#max_features_range = ['sqrt']
#max_depth_range = [10, 30]
#param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
#	'max_depth': max_depth_range}

# create a tree to train the models on
tree = RandomForestRegressor(criterion=criterion, verbose=verbose_tree, n_jobs=n_jobs,
	random_state=random_state, n_estimators=n_estimators_def,
	max_features=max_features_def, max_depth=max_depth_def)

# some feature selection
#print('selecting features...')
#print('input feature shape: ')
#print(X.shape)
#tree.fit(X, y)
#feature_select = SelectFromModel(tree, prefit=True, threshold=feature_threshold)
##print(tree.feature_importances_.sort())
#X_new = feature_select.transform(X)
#print('new input feature shape: ')
#print(X_new.shape)

# perform a grid search to tune the paramteres
#print('grid search to tune hyperparameters...')
#gs = GridSearchCV(estimator=tree,
#	param_grid=param_grid, scoring=None,
#	cv=n_folds, n_jobs=n_jobs, verbose=verbose_grid)
#gs = gs.fit(X_new, y)
#print(gs.scorer_)
#print('best score from grid search: %.3f' % gs.best_score_)
#print(gs.best_params_)
#best = gs.best_params_
#n_estimators_gs = best['n_estimators']
#max_depth_gs = best['max_depth']
#max_features_gs = best['max_features']

# run some cross validation
print('running cross validation to determine accuracy of model...')
scores = []
splits = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
for train, test in splits.split(X):
	tree.fit(X[train], y[train])
	predicted = tree.predict(X[test])
	score = mean_absolute_error(y[test], predicted)
	scores.append(score)
print(scores)

# determine which features to write to the file
n_estimators = n_estimators_def
max_depth = max_depth_def
max_features=max_features_def
score = np.mean(scores)

print('writing the data to file...')
params = (n_folds, n_estimators, max_depth, max_features, score)
write_hyperparams(params, hyperParamFile)
n_folds, n_estimators, max_depth, max_features, \
	score = read_hyperparams(hyperParamFile)
print(n_folds, n_estimators, max_depth, max_features, score)
