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


### Creating Models

Example of how to run the automated RandomForest model.

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
