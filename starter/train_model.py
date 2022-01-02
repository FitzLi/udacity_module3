# Script to train machine learning model.
import sys
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
# Add the necessary imports for the starter code.

CAT_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
NUM_FEATURES = [
    'age',
    'fnlgt', 
    'education_num', 
    'capital_gain', 
    'capital_loss', 
    'hours_per_week'
]
LABEL = "salary"

def compute_slice_performance(df_test, X_test, y_test, col, model):
    with open('../slice_output.txt', 'w') as file_out:
        file_out.write(f"Slice performance on column '{col}'\n")
        for val in df_test[col].unique():
            condi_slice = df_test[col] == val
            preds = inference(model, X_test[condi_slice])
            precision, recall, fbeta = compute_model_metrics(y_test[condi_slice], preds)
            file_out.write(f"Value: {val}, Precision: {precision.round(3)}, \
Recall: {recall.round(3)}, FBeta: {fbeta.round(3)}\n")

def train_and_evaluate():
    # Add code to load in the data.
    data = pd.read_csv("../data/census_cleaned.csv")

    data = data[NUM_FEATURES + CAT_FEATURES + [LABEL]]

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)


    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, numerical_features=NUM_FEATURES, label="salary", training=True
    )

    # Save encoder
    with open('../model/encoder.pkl', 'wb') as file_out:
        pickle.dump(encoder, file_out)

    # Save lb
    with open('../model/lb.pkl', 'wb') as file_out:
        pickle.dump(lb, file_out)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=CAT_FEATURES, numerical_features=NUM_FEATURES, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    with open('../model/model.pkl', 'wb') as file_out:
        pickle.dump(model, file_out)

    # Compute performance on entire test set
    preds_test = model.predict(X_test)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, preds_test)
    with open('../testset_performance.txt', 'w') as f_out:
        f_out.write(f"Performance on test dataset\nPrecision: {precision_test.round(3)}, Recall: {recall_test.round(3)}, FBeta: {fbeta_test.round(3)}")

    # Compute performance on column slice: 'education'
    compute_slice_performance(test, X_test, y_test, 'education', model)

if __name__ == '__main__':
    train_and_evaluate()
