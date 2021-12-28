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

# Evaluate model on 'education' slice
with open('../slice_output.txt', 'w') as file_out:
    for edu in test['education'].unique():
        condi_slice = test['education'] == edu
        preds = inference(model, X_test[condi_slice])
        precision, recall, fbeta = compute_model_metrics(y_test[condi_slice], preds)
        file_out.write(f"Education: {edu}, Precision: {precision.round(3)}, \
Recall: {recall.round(3)}, FBeta: {fbeta.round(3)}\n")
