# Tom Kawchak
# James Wang Research, Penn State University College of IST
# Titanic Kaggle Competition Submission

# import the necessary libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
from sklearn.preprocessing import Imputer
import sys
from sklearn.feature_selection import SelectFromModel

def write_hyperparams(hyperparams, fileName):
	f = open(fileName, 'w')
	f.write(' '.join(str(x) for x in hyperparams))
	f.close()

def read_hyperparams(fileName):
	f = open(fileName, 'r')
	n_estimators, max_depth, max_features = f.read().split()
	return int(n_estimators), int(max_depth), max_features

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
n_folds = 2
verbose_tree = 1
verbose_grid = 2
n_jobs = -1
random_state = 1
criterion = 'mse'

# read in the train dataset
print('loading training data...')
df = pd.read_csv('train.csv', header=0)

print('processing training data...')
X, y, ids = build_features(df)
#print(X)
#print(y)
#print(ids)

# create a Random Forest Classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print('creating a model...')
n_estimators_range = [10, 50]
max_features_range = ['sqrt']
max_depth_range = [10, 30]

# create a tree to train the models on
tree = RandomForestRegressor(criterion=criterion, verbose=verbose_tree, n_jobs=n_jobs,
	random_state=random_state, n_estimators=10, max_features=None, max_depth=None)
param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
	'max_depth': max_depth_range}

# some feature selection
print('selecting features...')
print('input feature shape: ')
print(X.shape)
tree.fit(X, y)
feature_select = SelectFromModel(tree, prefit=True, threshold=feature_threshold)
#print(tree.feature_importances_.sort())
X_new = feature_select.transform(X)
print('new input feature shape: ')
print(X_new.shape)

# perform a grid search to tune the paramteres
print('grid search to tune hyperparameters...')
gs = GridSearchCV(estimator=tree,
	param_grid=param_grid, scoring=None,
	cv=n_folds, n_jobs=n_jobs, verbose=verbose_grid)
gs = gs.fit(X_new, y)
print(gs.scorer_)
print('best score from grid search: %.3f' % gs.best_score_)
print(gs.best_params_)
best = gs.best_params_
n_estimators_gs = best['n_estimators']
max_depth_gs = best['max_depth']
max_features_gs = best['max_features']

print('writing the data to file...')
write_hyperparams((n_estimators_gs, max_depth_gs, max_features_gs), hyperParamFile)
print('completed')
