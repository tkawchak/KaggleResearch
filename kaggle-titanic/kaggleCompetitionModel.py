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


def build_features(df):
	# get rid of these columns
	df = df.drop(['PassengerId'], axis=1)

	# impute values that are missing
	imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

	# if a cabin value is know, assign 1, otherwise 0
	df['Cabin'] = df['Cabin'].replace(np.nan, '0')
	df['CabinVal'] = [1 if (x != '0') else 0 for x in df['Cabin']]
	df = df.drop(['Cabin'], axis=1)

	# leave only last names and count the number of entries with the same last name
	df['Name'] = [name.split()[0].replace(',', '') for name in df['Name'].values]
	df['NameCount'] = [df['Name'].values.tolist().count(x)-1 for x in df['Name'].values.tolist()]
	# drop the name column because we no longer use it.
	df = df.drop(['Name'], axis=1)

	# remove the leading letters from tickets.
	# call split and then take the last element in the returned list
	df['Ticket'] = df['Ticket'].replace(['LINE'], ['0'])
	df['Ticket'] = [ticket.split()[len(ticket.split())-1] for ticket in df['Ticket'].values]
	df['Ticket'] = df['Ticket'].astype(int)

	# map the sex label to 0, 1
	sex_mapping = {label:idx for idx, label in enumerate(np.unique(df['Sex']))}
	df['Sex'] = df['Sex'].map(sex_mapping)

	# map the embarkation point to 0, 1, 2
	df['Embarked'] = df['Embarked'].replace(np.nan, 'C')
	embark_mapping = {label:idx for idx, label in enumerate(np.unique(df['Embarked']))}
	df['Embarked'] = df['Embarked'].map(embark_mapping)
	df = df.drop(['Embarked'], axis=1)

	y = df['Survived'].values

	df = df.drop(['Survived'], axis=1)

	# impute missing values for the data
	imputer = imputer.fit(df)
	X = imputer.transform(df.values)

	return X, y


# define some constants
num_features = 9

# read in the train dataset
print('loading and processing training data...')
df = pd.read_csv('train.csv', header=0)
X, y = build_features(df)
#print(y)

# create a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

print('tuning hyperparameters with grid search...')
n_estimators_range = [10, 20, 30, 40, 50, 60]
max_features_range = ['sqrt']
max_depth_range = [4, 8, 12, 16]

# create a tree to train the models on
tree = RandomForestClassifier(criterion='entropy',
	random_state=1, n_jobs=-1)
param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
	'max_depth': max_depth_range}

# some initialization parameters for my implementation of a grid search
acc_max = 0
n_folds = 4
tree_max = RandomForestClassifier(criterion='entropy')
n_estimators_max = 0
max_features_max = 0
max_depth_max = 0
# splits the dataset in a stratified way
skf = StratifiedKFold(y, n_folds=n_folds, random_state=True, shuffle=True)
for n_estimators in n_estimators_range:
	for max_features in max_features_range:
		for max_depth in max_depth_range:
			#print(n_estimators, max_features, max_depth)
			acc_avg = 0
			acc_low = 1.0
			acc_high = 0.0
			for train_index, test_index in skf:
				tree = RandomForestClassifier(criterion='entropy',
        				random_state=1, n_jobs=-1, n_estimators=n_estimators,
					max_features=max_features, max_depth=max_depth)
				tree.fit(X[train_index], y[train_index])
				output = tree.predict(X[test_index])
				acc = accuracy_score(y[test_index], output)
				acc_avg = acc_avg + acc
				if acc > acc_high:
					acc_high = acc
				if acc < acc_low:
					acc_low = acc
			#print(acc_low, acc_high)
			acc_avg = acc_avg / float(n_folds)
			#print(acc_avg)
			if (acc_avg > acc_max):
				tree_max = tree
				acc_max = acc_avg
				n_estimators_max = n_estimators
				max_features_max = max_features
				max_depth_max = max_depth

print(acc_max, n_estimators_max, max_features_max, max_depth_max)
gs = GridSearchCV(estimator=tree,
	param_grid=param_grid,
	scoring='accuracy',
	cv=10, n_jobs=-1, verbose=1)
gs = gs.fit(X, y)
print('best score from grid search: %.3f' % gs.best_score_)
print(gs.best_params_)
best = gs.best_params_
n_estimators_gs = best['n_estimators']
max_depth_gs = best['max_depth']
max_features_gs = best['max_features']

# read in the test data and predict the outputs
print('reading and processing the test set data...')
df_all = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv', header=0)
df_all = df_all.append(df_test)

X, y = build_features(df_all)
train_X = X[:891, :]
test_X = X[891:, :]

train_y = y[:891]
#test_y = y[891:]

# train on all of the train data
print('training on all training data...')
tree_test = RandomForestClassifier(criterion='entropy',
	n_estimators=n_estimators_max, max_features=max_features_max,
	max_depth=max_depth_max, random_state=1, n_jobs=-1)
tree_test_gs = RandomForestClassifier(criterion='entropy',
	n_estimators=n_estimators_gs, max_features=max_features_gs,
	max_depth=max_depth_gs, random_state=1, n_jobs=-1)
tree_test.fit(train_X, train_y)
tree_test_gs.fit(train_X, train_y)

# predict the test set output
print('predicting the test output...')
test_pred = tree_test.predict(test_X).astype(int)
#test_pred = tree_test_gs.predict(test_X).astype(int)

# write output to a file
import csv
outputFile = 'titanicSurvivors.csv'
print('writing output to %s...' % outputFile)
prediction_file = open(outputFile, 'w')
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(['PassengerId', 'Survived'])
ids = list(range(892, 1310))
open_file_object.writerows(zip(ids, test_pred))
prediction_file.close()
print('completed')
