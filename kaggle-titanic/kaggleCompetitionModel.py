# Tom Kawchak
# James Wang Research, Penn State University College of IST
# Titanic Kaggle Competition Submission

# import the necessary libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble

# read in the train dataset
print('loading and processing training data...')
df = pd.read_csv('train.csv', header=0)

# get rid of these columns
df = df.drop(['Cabin', 'PassengerId'], axis=1)

# impute values that are missing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# drop entries with missing values
#df = df.dropna(how='any')
# SPLIT NAMES AND USE LEAVE ONE OUT METHOD
# TO FORM FEATURES
df['Name'] = [name.split()[0].replace(',', '') for name in df['Name'].values]
#print(df['Name'].values)
#from sklearn.model_selection import LeaveOneGroupOut
#logo = LeaveOneGroupOut()

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
test = df['Embarked'].isnull().values
#for i in test:
	#print(test[i])

#df['Embarked'] = df['Embarked'].replace(None, 'Q')
#df['Embarked'][df.Embarked.isnull()] = df['Embarked'].dropna().mode().values
df['Embarked'] = df['Embarked'].replace(np.nan, 'S')
embark_mapping = {label:idx for idx, label in enumerate(np.unique(df['Embarked']))}
df['Embarked'] = df['Embarked'].map(embark_mapping)

# get fare and age data to plot against survival
#print('making plot...')
#df_new = df[['Age', 'Pclass', 'Survived']]  # select subset of data
#df0 = df_new.loc[df_new['Survived'] == 0]  # get those that did not survive
#df1 = df_new.loc[df_new['Survived'] == 1]
#imputer0 = imputer.fit(df0)
#imputer1 = imputer.fit(df1)
#imputed_data0 = imputer.transform(df0.values)
#imputed_data1 = imputer.transform(df1.values)
#
#X0 = df0[['Age', 'Pclass']].values
#y0 = df0.iloc[:, 1].values
#X1 = df1[['Age', 'Pclass']].values
#y1 = df1.iloc[:, 1].values
#y = df.iloc[:, 1].values
#
#plt.scatter(X0[:, 0], X0[:, 1], color='red', marker='o', label='dead')
#plt.scatter(X1[:, 0], X1[:, 1], color='blue', marker='+', label='alive')
#plt.xlabel('Passenger Age')
#plt.ylabel('Passenger class')
#plt.title('Survival vs. Passenger Age and Class')
#plt.legend(loc='best')
#plt.show()

# create a Random Forest Classifier
#print('creating and training classifier on 75% of data...')
from sklearn.ensemble import RandomForestClassifier
#tree1 = RandomForestClassifier(criterion='entropy',
#	n_estimators=10, random_state=1, n_jobs=-1)
#tree2 = RandomForestClassifier(criterion='entropy',
#	n_estimators=8, random_state=1, n_jobs=-1)
#tree3 = RandomForestClassifier(criterion='entropy',
#	n_estimators=6, random_state=1, n_jobs=-1)
#tree4 = RandomForestClassifier(criterion='entropy',
#	n_estimators=12, random_state=1, n_jobs=-1)
#tree5 = RandomForestClassifier(criterion='entropy',
#	n_estimators=14, random_state=1, n_jobs=-1)
#tree6 = RandomForestClassifier(criterion='entropy',
#	n_estimators=16, random_state=1, n_jobs=-1)


# impute missing values for the data
imputer_training = imputer.fit(df)
train_data = imputer_training.transform(df.values)
# to test out the algorithm on this data I should split it
# into train and test data to evaluate performance
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV

#stratKFolds = StratifiedKFold(y=train_data[:, 0], n_folds=5, random_state=1, shuffle=True)
#for k, (train_set, test_set) in enumerate(stratKFolds):
print('tuning hyperparameters...')
n_estimators_range = [1, 5, 10, 15, 20, 25, 30, 35, 40]
max_features_range = [int(x) for x in range(1, 9)]
max_depth_range = [2 ,4, 6, 8, 10]
#print(n_estimators_range.values())
tree = RandomForestClassifier(criterion='entropy',
	random_state=1, n_jobs=-1)
param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range,
	'max_depth': max_depth_range}
rs = RandomizedSearchCV(estimator=tree, 
	param_distributions=param_grid,
	scoring='accuracy',
	cv=5, n_jobs=-1)
rs = rs.fit(train_data[:, 1:9], train_data[:, 0])
print(rs.best_score_)
print(rs.best_params_)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train_data[:, 1:9], train_data[:, 0],
#	test_size = 0.25, random_state=1)
#tree1.fit(X_train, y_train)
#tree2.fit(X_train, y_train)
#tree3.fit(X_train, y_train)
#tree4.fit(X_train, y_train)
#tree5.fit(X_train, y_train)
#tree6.fit(X_train, y_train)
#predicted1 = tree1.predict(X_test)
#predicted2 = tree2.predict(X_test)
#predicted3 = tree3.predict(X_test)
#predicted4 = tree4.predict(X_test)
#predicted5 = tree5.predict(X_test)
#predicted6 = tree6.predict(X_test)
#print(predicted)

# determine how good of a job the model does
from sklearn.metrics import accuracy_score
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted1)))
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted2)))
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted3)))
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted4)))
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted5)))
#print('accuracy of model on test set of training data: %.3f' 
#	% (accuracy_score(y_true=y_test, y_pred=predicted6)))

print('training on all training data...')
tree_test = RandomForestClassifier(criterion='entropy',
	n_estimators=25, max_features=7, max_depth=10, random_state=1, n_jobs=-1)
tree_test.fit(train_data[:, 1:9], train_data[:, 0])

# read in the test data and predict the outputs
print('reading and processing the input data...')
df_test = pd.read_csv('test.csv', header=0)
ids = df_test['PassengerId'].values
df_test = df_test.drop(['Name', 'Cabin', 'PassengerId'], axis=1)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)
df_test['Ticket'] = df_test['Ticket'].replace(['LINE'], ['0'])
df_test['Ticket'] = [ticket.split()[len(ticket.split())-1] for ticket in df_test['Ticket'].values]
df_test['Ticket'] = df_test['Ticket'].astype(int)
df_test['Embarked'] = df_test['Embarked'].replace(np.nan, 'S')
df_test['Embarked'] = df_test['Embarked'].map(embark_mapping)
imputer2 = imputer.fit(df_test)
test_data = imputer2.transform(df_test.values)
print('predicting the output of the test data...')
#test_pred = tree.predict(test_data).astype(int)
test_pred = tree_test.predict(test_data).astype(int)

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
