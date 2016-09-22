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
df = df.drop(['Name', 'Cabin', 'PassengerId'], axis=1)

# impute values that are missing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# drop entries with missing values
#df = df.dropna(how='any')

# remove the leading letters from tickets.
# call split and then take the last element in the returned list
df['Ticket'] = df['Ticket'].replace(['LINE'], ['0'])
df['Ticket'] = [ticket.split()[len(ticket.split())-1] for ticket in df['Ticket'].values]
df['Ticket'] = df['Ticket'].astype(int)

# map the sex label to 0, 1
sex_mapping = {label:idx for idx, label in enumerate(np.unique(df['Sex']))}
df['Sex'] = df['Sex'].map(sex_mapping)

# map the embarkation point to 0, 1, 2
df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
embark_mapping = {label:idx for idx, label in enumerate(np.unique(df['Embarked']))}
df['Embarked'] = df['Embarked'].map(embark_mapping)

# get fare and age data to plot against survival
print('making plot...')
df_new = df[['Age', 'Pclass', 'Survived']]  # select subset of data
df0 = df_new.loc[df_new['Survived'] == 0]  # get those that did not survive
df1 = df_new.loc[df_new['Survived'] == 1]
imputer0 = imputer.fit(df0)
imputer1 = imputer.fit(df1)
imputed_data0 = imputer.transform(df0.values)
imputed_data1 = imputer.transform(df1.values)

X0 = df0[['Age', 'Pclass']].values
y0 = df0.iloc[:, 1].values
X1 = df1[['Age', 'Pclass']].values
y1 = df1.iloc[:, 1].values
y = df.iloc[:, 1].values

#plt.scatter(X0[:, 0], X0[:, 1], color='red', marker='o', label='dead')
#plt.scatter(X1[:, 0], X1[:, 1], color='blue', marker='+', label='alive')
#plt.xlabel('Passenger Age')
#plt.ylabel('Passenger class')
#plt.title('Survival vs. Passenger Age and Class')
#plt.legend(loc='best')
#plt.show()

# create a Random Forest Classifier
print('creating and training classifier...')
from sklearn.ensemble import RandomForestClassifier
tree = RandomForestClassifier(criterion='entropy',
	n_estimators=10, random_state=1, n_jobs=-1)

# impute missing values for the data
imputer_training = imputer.fit(df)
train_data = imputer_training.transform(df.values)
# to test out the algorithm on this data I should split it
# into train and test data to evaluate performance
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data[:, 1:9], train_data[:, 0],
	test_size = 0.25, random_state=1)
tree.fit(X_train, y_train)
predicted = tree.predict(X_test)
#print(predicted)

# determine how good of a job the model does
from sklearn.metrics import accuracy_score
print('accuracy of model on training data: %.3f' 
	% (accuracy_score(y_true=y_test, y_pred=predicted)))

import sys
sys.exit('quiting....')

# read in the test data and predict the outputs
print('reading and processing the input data...')
df_test = pd.read_csv('test.csv', header=0)
ids = df_test['PassengerId'].values
df_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)
df_test.Embarked[df_test.Embarked.isnull()] = df_test.Embarked.dropna().mode().values
df_test['Embarked'] = df_test['Embarked'].map(embark_mapping)
imputer2 = imputer.fit(df_test)
test_data = imputer2.transform(df_test.values)
print('predicting the output of the test data...')
test_pred = tree.predict(test_data).astype(int)

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
