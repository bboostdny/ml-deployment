import pandas as pd

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(X)


def compute_metrics_on_slices_categorical(df, cat_features, metrics_output_path='../../model/metrics_slices.txt'):
    """
    Computes metrics (precision, recall, fbeta) on slices of categorical columns.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features, predictions and ground truth
    cat_features
        List of categorical column names
    Returns
    -------

    """
    df = df.fillna('Empty')
    df_metrics = pd.DataFrame(columns=['cat_feature', 'slice', 'precision',
                                       'recall', 'fbeta', 'support'])

    precision, recall, fbeta = compute_model_metrics(df['y_true'], df['y_pred'])
    df_metrics.loc[len(df_metrics)] = ['overall', 'overall', round(precision, 4),
                                       round(recall, 4), round(fbeta, 4), df.shape[0]]

    for cat_feature in cat_features:
        # Retrieve list of possible slices
        slices = df[cat_feature].unique()
        for slice in slices:
            df_tmp = df[df[cat_feature] == slice]
            # Calculate metrics for a slice
            precision, recall, fbeta = compute_model_metrics(df_tmp['y_true'], df_tmp['y_pred'])
            support = df_tmp.shape[0]
            df_metrics.loc[len(df_metrics)] = [cat_feature, slice, round(precision, 4),
                                               round(recall, 4), round(fbeta, 4), support]

    df_metrics.to_csv(metrics_output_path, sep='\t', index=False)
    print(f'Metrics calculated on slices of categorical columns have been saved to {metrics_output_path}')
