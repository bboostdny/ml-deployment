# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model has been developed by Michal G. for Udacity Nanodegree course. Model has been trained
in January 2024. Model used is Random Forest Classifier with the default hyperparameters using
scikit-learn in version 1.0.1.
## Intended Use
The model should be used to predict whether the income of a person exceeds $50k/year.
The model could be used for credit scoring.
## Training Data
The data was obtained from a Census database. The dataset contains 32k records and a 80-20 split was used to break this into a train and test set. No stratification was done. 
To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.
## Evaluation Data
Model has been evaluated on a subset of training dataset extracted in training process.
## Metrics
The model was evaluated using F1 score. The value is 0.6744.

## Ethical Considerations
The model considers only a part of aspects that impact the income level of a particular person and therefore should not be used as 
standalone tool to assess the income level. 
## Caveats and Recommendations
The training data has different number of datapoints for particular groups of people and therefore does not perform equally well
in all of the data slices.