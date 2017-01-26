import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import dump_svmlight_file

# define some constants
n_jobs = 3
num_features_cat = 20
num_features_cont = 10
data_dir = '../data/'
xgboost_data_dir = '../xgboost_data/'

# read in the train dataset
print('loading data...')
df_cont = pd.read_csv(data_dir+'continuous_all.csv', header=0, index_col=0)
df_cat = pd.read_csv(data_dir+'categorical_all.csv', header=0, index_col=0)
df_y = pd.read_csv(data_dir+'y_all.csv', header=0, index_col=0)

df_test_cont = pd.read_csv(data_dir+'continuous_test_all.csv', header=0, index_col=0)
df_test_cat = pd.read_csv(data_dir+'categorical_test_all.csv', header=0, index_col=0)

X_cont = df_cont.values
X_cat = df_cat.values
y = np.ravel(df_y.values)

# create a Random Forest Classifier
print('creating a model...')
# create a tree to select features
tree_cat = RandomForestRegressor(n_jobs=n_jobs,
	random_state=1, n_estimators=10,
	max_features='sqrt', max_depth=10)
tree_cont = RandomForestRegressor(n_jobs=n_jobs,
	random_state=1, n_estimators=10,
	max_features='sqrt', max_depth=10)

# some feature selection
print('selecting features...')

# use variance threshold to select features
# many of the features are in categories with few vars
selector_variance_cat = VarianceThreshold(threshold=0.1)
X_cat = selector_variance_cat.fit_transform(X_cat)
print('shape of X_cat after variance threshold')
print(X_cat.shape)

# create a basic tree for continuous features
print('fitting tree to continuous data...')
tree_cont.fit(X_cont, y)
feature_importances_cont = tree_cont.feature_importances_
feature_mapping_cont = {importance:idx for idx, importance in \
	enumerate(feature_importances_cont)}
sorted_features_cont = feature_importances_cont.argsort()
sorted_indices_cont = []
print(sorted_features_cont)
for x in sorted_features_cont[:num_features_cont]:
	sorted_indices_cont.insert(0, x)

# create a basic tree for categorical features
print('fitting tree to categorical data...')
tree_cat.fit(X_cat, y)
feature_importances_cat = tree_cat.feature_importances_
feature_mapping_cat = {importance:idx for idx, importance in \
	enumerate(feature_importances_cat)}
sorted_features_cat = feature_importances_cat.argsort()
sorted_indices_cat = []
print(sorted_features_cat)
for x in sorted_features_cat[:num_features_cat]:
	sorted_indices_cat.insert(0, x)

	
print('writing output data...')
df_cont_new = df_cont.iloc[:, sorted_indices_cont[:num_features_cont]]
df_cont_new.to_csv(path_or_buf=xgboost_data_dir+'continuous_selected.csv')

df_cat_new = df_cat.iloc[:, sorted_indices_cat[:num_features_cat]]
df_cat_new.to_csv(path_or_buf=xgboost_data_dir+'categorical_selected.csv')

df_test_cont_new = df_test_cont.iloc[:, sorted_indices_cont[:num_features_cont]]
df_test_cont_new.to_csv(path_or_buf=xgboost_data_dir+'continuous_test_selected.csv')
df_test_cat_new = df_test_cat.iloc[:, sorted_indices_cat[:num_features_cat]]
df_test_cat_new.to_csv(path_or_buf=xgboost_data_dir+'categorical_test_selected.csv')

X_cont_new = df_cont_new.values
dump_svmlight_file(X_cont_new, y, xgboost_data_dir+'continuous_selected.dat',
	zero_based=True, multilabel=False)

X_test_cont_new = df_test_cont_new.values
y_test = [ 0 for x in range(X_test_cont_new.shape[0])]
dump_svmlight_file(X_test_cont_new, y_test, xgboost_data_dir+'continuous_test_selected.dat',
	zero_based=True, multilabel=False)

X_cat_new = df_cat_new.values
print('df cat new head')
print(df_cat_new.head())
print('df cat new sum')
print(df_cat_new.sum(axis=0))
print('df y head')
print(df_y.head())
dump_svmlight_file(X_cat_new, y, xgboost_data_dir+'categorical_selected.dat',
	zero_based=True, multilabel=False)

X_test_cat_new = df_test_cat_new.values
print('df cat new head')
print(df_cat_new.head())
dump_svmlight_file(X_test_cat_new, y_test, xgboost_data_dir+'categorical_test_selected.dat',
	zero_based=True, multilabel=False)

print('successfully wrote selected features to file!')
