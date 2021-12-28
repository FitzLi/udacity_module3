# Script to train machine learning model.
import sys
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(sys.argv[1])
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

# Save encoder
with open('../model/encoder.pkl', 'wb') as file_out:
    pickle.dump(encoder, file_out)

# Save lb
with open('../model/lb.pkl', 'wb') as file_out:
    pickle.dump(lb, file_out)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'wb') as file_out:
    pickle.dump(model, file_out)
