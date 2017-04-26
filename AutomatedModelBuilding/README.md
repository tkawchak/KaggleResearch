# Automated Model Builder

This contains a framework (still under development) for automated machine learning model construction in Python (3.5).
Right now, the implemented models are in iPython notebooks with example code on how to run: 
* [DataProcessing.ipynb](DataProcessing.ipynb) Preprocesses data according to the feature description file given.
* [SVM.ipynb](SVM.ipynb), [RandomForest.ipynb](RandomForest.ipynb), [XGBoost.ipynb](XGBoost.ipynb) Implement the different learning models.
* [ModelBuilder](ModelBuilder.ipynb) can be used for simple ensembling for the models.

### Models Available
* Random Forest
* XGBoost
* SVM

## Dependencies

* [Sklearn](http://scikit-learn.org/stable/index.html)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Pandas](http://pandas.pydata.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Jupyter Notebooks](https://jupyter.org/) (for running the notebooks)

## Getting Started

### Formatting Data Files

Data must be separated into these files:
* train_X.csv
* train_y.csv
* train_ids.csv
* test_X.csv
* test_ids.csv
* feature_descriptions.csv

For more details, see the example [feature description file](FeatureDescriptions.xlsx)

### Available preprocessing functions:

Impute values
* UNIQ - use a unique value
* MEAN - use the mean
* MED - use the median
* MODE - use the mode
* <number> use <number> 
* DEL - delete column

Categorical Encoding
* MAP - map to integers
* OHE - perform one hot encoding

Continuous Encoding
* NRM1 - divide data by the 1-norm
* SCL1 - scale data to range (-1, 1)
* THRS - threshold image data (0 if < 20, 1 if > 20)


## Examples

See the ipython notebooks for more details on using the model builder.

### Data Preprocessing

#### Example of how to use DataPreprocessing class.

```
train_data = data_filepath+'houseprices_data_train.csv'
train_labels = data_filepath+'houseprices_labels_train.csv'
train_ids = data_filepath+'houseprices_ids_train.csv'
test_data = data_filepath+'houseprices_data_test.csv'
test_ids = data_filepath+'houseprices_ids_test.csv'
description = data_filepath+'houseprices_feature_descriptions.csv'

proc = Preprocessor(train_data_file=train_data,
                 train_label_file=train_labels,
                 train_ids_file=train_ids,
                 test_data_file=test_data,
                 test_ids_file=test_ids,
                 instr_file=description)
proc.read_data()
proc.process()
proc.write_data()
```

### Creating Models

#### Example of how to run the automated RandomForest model.
Other models have a similar interface.

```
model = RandomForest(n_jobs=2, regressor=True, criterion='mse', 
                    opt_func=log_e, inv_opt_func=exp_e, scorer=neg_rmse,
                    n_estimators=100)
model.read_data()
features = model.select_features_2(score_func_name='f_regression', percentage=100)
model.tune_params()
model.train_model(n_estimators=10000)
pred = model.predict_output()
model.write_output(output_file='houseprices_output_RandomForest.csv', header=['Id', 'SalesPrice'])
```

### Model Builder

#### Example of how to use the output of different models to form an ensemble

```
builder = ModelBuilder()
#builder.majority_output()
builder.average_output()
builder.write_output(output_file='houseprices_output_avg.csv', header=['Id', 'SalePrice'])
```