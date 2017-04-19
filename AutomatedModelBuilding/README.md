# Automated Model Builder

This contains a framework (still under development) for automated machine learning model construction.

## Current capabilities

### Preprocess Data

### Train Models
* Random Forest
* XGBoost

### Ensembling still needs to be implemented

## Dependencies
[Sklearn](http://scikit-learn.org/stable/index.html)
[XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)
[Pandas](http://pandas.pydata.org/)
[Numpy](http://www.numpy.org/)
[Scipy](https://www.scipy.org/)

## Getting Started

Information on how to run the Automated Model Builder.

### Available preprocessing functions:
* OHE
* UNIQ
* 

For more details, see the example [feature description file](FeatureDescriptions.xlsx)


### Creating Models

```
model = RandomForest()
model.read_data()
model.select_features()
model.select_parameters()
model.train()
model.fit()
```
