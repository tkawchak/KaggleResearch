# Tom Kawchak
# James Wang Research, Penn State University College of IST
# Titanic Kaggle Competition Submission

# import the necessary libraries
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
from sklearn.preprocessing import Imputer
import sys
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def write_hyperparams(hyperparams, fileName):
	f = open(fileName, 'a')
	f.write(' '.join(str(x) for x in hyperparams) + '\n')
	f.close()

def read_hyperparams(fileName):
	f = open(fileName, 'r')
	n_folds, n_estimators, max_depth, max_features, \
		score = f.read().split()
	return int(n_folds), int(n_estimators), int(max_depth), \
		max_features, float(score)

def build_features(df):
	''' build the features to use for the model'''
	
	# remove the labels and the ids for use
	y = df.pop('loss').values
	ids = df.pop('id').values

	# get rid of the columns that don't give us useful information


	# create an imputer for imputer values later
	imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
	
	# determine whether data is categorical or continuous
	for column in df:
		#print(column)
		if 'cat' in column:
			# create an encoding for categorical vars
			mapping = {label:idx for idx, label in \
				enumerate(np.unique(df[column]))}
			#print(mapping)
			df[column] = df[column].map(mapping)
			
		else:
			# do nothing here
			a = 1

	imputer = imputer.fit(df)
	X = imputer.transform(df.values)
	return X, y, ids


# define some constants
hyperParamFile = 'hyperparams.txt'
feature_threshold = 0.005
n_folds = 3
verbose_tree = 2
verbose_grid = 0
n_jobs = 2
random_state = 1
criterion = 'mae'

# read in the train dataset
print('loading training data...')
df = pd.read_csv('train.csv', header=0)

print('processing training data...')
X, y, ids = build_features(df)
#print(X)
#print(y)
#print(ids)

# create a Random Forest Classifier
print('creating a model...')
n_estimators_def = 10
max_features_def = 'sqrt'
max_depth_def = 10
n_estimators_range = [10, 50]
max_features_range = ['sqrt']
max_depth_range = [10, 30]
param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
	'max_depth': max_depth_range}

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
	print(X[train].shape)
	print(y[train].shape)
	tree.fit(X[train], y[train])
	predicted = tree.predict(X[test])
	score = mean_absolute_error(y[test], predicted)
	scores.append(score)
	print(score)
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



sys.exit('completed')

# read in the test data and predict the outputs
print('reading the whole data set...')
df_all = pd.read_csv('train.csv', header=0)
train_rows, train_cols = df_all.shape
#print(train_rows, train_cols)
df_test = pd.read_csv('test.csv', header=0)
test_rows, test_cols = df_test.shape
#print(test_rows, test_cols)
df_all = df_all.append(df_test)
#print(df_all.shape)

# process all of the training data and split using
# the feature selection used before
print('processing the whole data set...')
X, y, ids = build_features(df_all)
X_new = feature_select.transform(X)
print(X_new.shape)
train_X = X_new[:train_rows, :]
test_X = X_new[train_rows:, :]

train_y = y[:train_rows]
#test_y = y[train_rows:]

ids_train = ids[:train_rows]
ids_test = ids[train_rows:]

# train on all of the train data
print('training on all training data...')
tree_test_gs = RandomForestRegressor(criterion=criterion,
	n_estimators=n_estimators_gs, max_features=max_features_gs,
	max_depth=max_depth_gs, random_state=random_state, n_jobs=n_jobs)
tree_test_gs.fit(train_X, train_y)

# predict the test set output
print('predicting the test output...')
test_pred = tree_test_gs.predict(test_X).astype(float)

# write output to a file
import csv
outputFile = 'allstateClaims.csv'
print('writing output to %s...' % outputFile)
prediction_file = open(outputFile, 'w')
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(['id', 'loss'])
open_file_object.writerows(zip(ids_test, test_pred))
prediction_file.close()
print('completed')
