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

def write_hyperparams(hyperparams, fileName):
	f = open(fileName, 'w')
	f.write(' '.join(str(x) for x in hyperparams))
	f.close()

def read_hyperparams(fileName):
	f = open(fileName, 'r')
	n_folds, n_estimators, max_depth, max_features, \
		 score = f.read().split()
	return int(n_fodls), int(n_estimators), int(max_depth), \
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
verbose_tree = 2
n_jobs = 2
random_state = 1
criterion = 'mae'

print('reading parameters from file...')
#n_estimators_gs, max_depth_gs, max_features_gs = read_hyperparams(hyperParamFile)
n_estimators_gs = 10
max_depth_gs = 10
max_features_gs = 'sqrt'

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
#X_new = feature_select.transform(X)
#print(X_new.shape)
train_X = X[:train_rows, :]
test_X = X[train_rows:, :]

train_y = y[:train_rows]
#test_y = y[train_rows:]

ids_train = ids[:train_rows]
ids_test = ids[train_rows:]

# train on all of the train data
print('training on all training data...')
tree_test_gs = RandomForestRegressor(criterion=criterion,
	n_estimators=n_estimators_gs, max_features=max_features_gs,
	max_depth=max_depth_gs, random_state=random_state, n_jobs=n_jobs,
	verbose=verbose_tree)
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
