# Tom Kawchak
# James Wang Research, Penn State University College of IST
# Titanic Kaggle Competition Submission

# import the necessary libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble

num_features = 10

# read in the train dataset
print('loading and processing training data...')
df = pd.read_csv('train.csv', header=0)

# get rid of these columns
df = df.drop(['PassengerId'], axis=1)

# impute values that are missing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# if a cabin value is know, assign 1, otherwise 0
df['Cabin'] = df['Cabin'].replace(np.nan, '0')
df['CabinVal'] = [1 if (x != '0') else 0 for x in df['Cabin']]
df = df.drop(['Cabin'], axis=1)

# leave only last names and count the number of entries with the same last name
df['Name'] = [name.split()[0].replace(',', '') for name in df['Name'].values]
df['NameCount'] = [df['Name'].values.tolist().count(x) for x in df['Name'].values.tolist()]
# drop the name column because we no longer use it.
# what other uses could we have???
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

# create a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# impute missing values for the data
imputer_training = imputer.fit(df)
train_data = imputer_training.transform(df.values)
# to test out the algorithm on this data I should split it
# into train and test data to evaluate performance
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

#stratKFolds = StratifiedKFold(y=train_data[:, 0], n_folds=5, random_state=1, shuffle=True)
#for k, (train_set, test_set) in enumerate(stratKFolds):
print('tuning hyperparameters...')
n_estimators_range = [10, 20, 30, 40, 50, 60]
max_features_range = ['sqrt']
max_depth_range = [4, 8, 12, 16]
#print(n_estimators_range.values())
tree = RandomForestClassifier(criterion='entropy',
	random_state=1, n_jobs=-1)
param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
	'max_depth': max_depth_range}
#rs = RandomizedSearchCV(estimator=tree, 
#	param_distributions=param_grid,
#	scoring='accuracy',
#	cv=3, n_jobs=-1)
#rs = rs.fit(train_data[:, 1:num_features+1], train_data[:, 0])
#print(rs.best_score_)
#print(rs.best_params_)

# figure out how to iterate over all of these without
# four nested for loops
X = train_data[:, 1:num_features+1]
y = train_data[:, 0]
acc_max = 0
n_folds = 4
n_estimators_max = 0
max_features_max = 0
max_depth_max = 0
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
skf = StratifiedKFold(y, n_folds=n_folds, random_state=True, shuffle=True)
for n_estimators in n_estimators_range:
	for max_features in max_features_range:
		for max_depth in max_depth_range:
			print(n_estimators, max_features, max_depth)
			acc_avg = 0
			for train_index, test_index in skf:
				tree = RandomForestClassifier(criterion='entropy',
        				random_state=1, n_jobs=-1, n_estimators=n_estimators,
					max_features=max_features, max_depth=max_depth)
				tree.fit(X[train_index], y[train_index])
				output = tree.predict(X[test_index])
				acc = accuracy_score(y[test_index], output)
				acc_avg = acc_avg + acc
			acc_avg = acc_avg / float(n_folds)
			if (acc_avg > acc_max):
				acc_max = acc_avg
				n_estimators_max = n_estimators
				max_features_max = max_features
				max_depth_max = max_depth

print(acc_max, n_estimators_max, max_features_max, max_depth_max)
			
#gs = GridSearchCV(estimator=tree,
#	param_grid=param_grid,
#	scoring='accuracy',
#	cv=10, n_jobs=-1, verbose=1)
#gs = gs.fit(train_data[:, 1:num_features+1], train_data[:, 0])
#print('best score from grid search: %.3f' % gs.best_score_)
#print(gs.best_params_)
#best = gs.best_params_
#n_estimators = best['n_estimators']
#max_depth = best['max_depth']
#max_features = best['max_features']

# determine how good of a job the model does
print('training on all training data...')
tree_test = RandomForestClassifier(criterion='entropy',
	n_estimators=n_estimators_max, max_features=max_features_max,
		 max_depth=max_depth_max, random_state=1, n_jobs=-1)
tree_test.fit(train_data[:, 1:num_features+1], train_data[:, 0])

# read in the test data and predict the outputs
print('reading and processing the input data...')
df_test = pd.read_csv('test.csv', header=0)
ids = df_test['PassengerId'].values
df_test = df_test.drop(['PassengerId'], axis=1)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)
df_test['Ticket'] = df_test['Ticket'].replace(['LINE'], ['0'])
df_test['Ticket'] = [ticket.split()[len(ticket.split())-1] for ticket in df_test['Ticket'].values]
df_test['Ticket'] = df_test['Ticket'].astype(int)
df_test['Embarked'] = df_test['Embarked'].replace(np.nan, 'S')
df_test['Embarked'] = df_test['Embarked'].map(embark_mapping)
df_test['Name'] = [name.split()[0].replace(',', '') for name in df_test['Name'].values]
df_test['NameCount'] = [df_test['Name'].values.tolist().count(x) for x in df_test['Name'].values.tolist()]
df_test['Cabin'] = df_test['Cabin'].replace(np.nan, '0')
df_test['CabinVal'] = [1 if (x != '0') else 0 for x in df_test['Cabin']]
df_test = df_test.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
imputer2 = imputer.fit(df_test)
test_data = imputer2.transform(df_test.values)
print('predicting the output of the test data...')
test_pred = tree_test.predict(test_data).astype(int)
#test_pred = tree.predict(test_data).astype(int)
#test_pred = gs.predict(test_data).astype(int)

# write output to a file
import csv
outputFile = 'titanicSurvivors.csv'
print('writing output to %s...' % outputFile)
prediction_file = open(outputFile, 'w')
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(zip(ids, test_pred))
prediction_file.close()
print('completed')
