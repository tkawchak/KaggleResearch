import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

def build_features(df):
    ''' build the features to use for the model'''
    
    # remove the labels and the ids for use
    
    y = df.pop('loss').values if 'loss' in list(df) else None
    ids = df.pop('id').values

    # create an imputer for imputer values later
    imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    
    # determine whether data is categorical or continuous
    leave_one_out_cols = []
    leve_one_out_counts = []
    one_hot_cols = []
    col_count = 0
    for column in df:

        # determine if the column is categorical or not
        if 'cat' in column:

            # create an encoding for categorical vars
            mapping = {label:idx for idx, label in \
                enumerate(np.unique(df[column]))}

            # convert everything into integers for categorical
            df[column] = df[column].map(mapping)
            df[column] = df[column].astype(int)
            
            unique_elems = len(mapping)

            # perform leave-one-out counting
            if unique_elems > 10:
                #df[column] = [df[column].values.tolist().count(x)-1 for \
                #	x in df[column].values.tolist()]
                leave_one_out_cols.append(col_count)
                # initialize counts to -1 for Leave One Out counting
                leave_one_out_counts.append([-1 for x in range(unique_elems)])
            else:
                one_hot_cols.append(col_count)

        col_count += 1

    imputer = imputer.fit(df)
    X = imputer.transform(df.values)
    X_cat = X[:, :116]
    X_cont = X[:, 116:]

    # transform data to leave-one-out counting
    for num, col in enumerate(leave_one_out_cols):
        # count the data
        for idx, value in enumerate(X_cat[:, col]):
            leave_one_out_counts[num][int(value)] = leave_one_out_counts[num][int(value)] + 1
        # apply the counted data to form LOO data
        for idx, value in enumerate(X_cat[:, col]):
            X_cat[idx][col] = leave_one_out_counts[num][int(value)] 
            
    # transform data to one-hot encoded
    one_hot_encoder = OneHotEncoder(categorical_features=one_hot_cols, sparse=False)
    X_cat = one_hot_encoder.fit_transform(X_cat)
    
    print('X cat size')
    print(X_cat.shape)
    print('X cont size')
    print(X_cont.shape)
    
    return X_cat, X_cont, y, ids


# define some constants
n_jobs = 3
data_dir = '../data/'

# read in the train dataset
print('loading data...')
df_train = pd.read_csv(data_dir + 'train.csv', header=0)
df_test = pd.read_csv(data_dir + 'test.csv', header=0)
# make sure this does what I want
df_test = df_train.append(df_test)

print('processing training data...')
X_cat, X_cont, y, ids = build_features(df_train)
print(len(ids))
print('processing test data...')
X_test_cat, X_test_cont, y_test, ids_test = build_features(df_test)
print(len(ids_test))

df_cont = pd.DataFrame(X_cont)
df_cat = pd.DataFrame(X_cat)
df_y = pd.DataFrame(y)
df_ids = pd.DataFrame(ids)

df_test_cont = pd.DataFrame(X_test_cont)
df_test_cat = pd.DataFrame(X_test_cat)
df_test_y = pd.DataFrame(y_test)
df_test_ids = pd.DataFrame(ids_test)

print('writing the data to .csv files...')
df_cont.to_csv(path_or_buf=data_dir+'continuous_all.csv')
df_cat.to_csv(path_or_buf=data_dir+'categorical_all.csv')
df_y.to_csv(path_or_buf=data_dir+'y_all.csv')
df_ids.to_csv(path_or_buf=data_dir+'ids_all.csv')

df_test_cont.to_csv(path_or_buf=data_dir+'continuous_test_all.csv')
df_test_cat.to_csv(path_or_buf=data_dir+'categorical_test_all.csv')
df_test_ids.to_csv(path_or_buf=data_dir+'ids_test_all.csv')

print('data has been processed and written to .csv files in ' + data_dir)
