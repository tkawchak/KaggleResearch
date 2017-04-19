# Automated Model Builder

This contains a framework (still under development) for automated machine learning model construction.

## Dependencies
[Sklearn]
[XGBoost]
[Pandas]
[Numbpy]
[Scipy]

## Getting Started

### Available preprocessing functions:
* OHE
* UNIQ
* 

For more details, see the example [feature description file]


### Creating Models

'''
model = RandomForest()
model.read_data()
model.select_features()
model.select_parameters()
model.train()
model.fit()
'''
