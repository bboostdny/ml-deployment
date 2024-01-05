# Script to train machine learning model.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model, inference, compute_metrics_on_slices_categorical

# Add the necessary imports for the starter code.
# Add code to load in the data.
input_data_path = '../data/census.csv'
model_path = '../model/'
metrics_output_path = '../model/metrics_slices.txt'
data = pd.read_csv(input_data_path)
data = data.drop("fnlgt", axis=1)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)

# Save trained model
print(f'Saving trained model to: {model_path}')
with open(os.path.join(model_path, 'model.pkl'), 'wb') as file:
    pickle.dump(model, file)

with open(os.path.join(model_path, 'encoder.pkl'), 'wb') as file:
    pickle.dump(encoder, file)

with open(os.path.join(model_path, 'lb.pkl'), 'wb') as file:
    pickle.dump(lb, file)

# Calculate metrics on data slices
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
test['y_true'] = y_test
predictions = inference(model, X_test)
test['y_pred'] = predictions
compute_metrics_on_slices_categorical(test, cat_features, metrics_output_path=metrics_output_path)
