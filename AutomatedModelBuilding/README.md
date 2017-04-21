# Automated Model Builder

This contains a framework (still under development) for automated machine learning model construction.

## Current capabilities

### Preprocess Data

### Train Models
* Random Forest
* XGBoost

### Ensembling still needs to be implemented

## Dependencies

* [Sklearn](http://scikit-learn.org/stable/index.html)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Pandas](http://pandas.pydata.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)

## Getting Started

Information on how to run the Automated Model Builder.

### Formatting Data Files

Data must be separated into these files:
* train_X.csv
* train_y.csv
* train_ids.csv
* test_X.csv
* test_ids.csv
* feature_descriptions.csv

### Available preprocessing functions:

Impute values
* UNIQ
* MEAN
* MED
* MODE
* (set to specific value)

Categorical Encoding
* MAP
* OHE

Continuous Encoding
* NRM1
* SCL1

For more details, see the example [feature description file](FeatureDescriptions.xlsx)


### Creating Models

This is an example of how to run the automated RandomForest model.

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
