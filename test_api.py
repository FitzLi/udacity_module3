import json
from fastapi.testclient import TestClient

from main import app, BASE_DIR, ENCODER_PATH

import pytest

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "What's up, man~~~!"

def test_low_salary_inference(client):
    payload = {
        "workclass": "state_gov",
        "education": "bachelors",
        "marital_status": "never_married",
        "occupation": "adm_clerical",
        "relationship": "not_in_family",
        "race": "white",
        "sex": "male",
        "native_country": "united_states",
        "age": 39,
        "fnlgt": 77516,
        "education_num": 13,
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40
    }
    r = client.post("/inference/", json=payload)
    assert r.json()["Predicted salary"] == "<=50k"

def test_high_salary_inference(client):
    payload = {
        "workclass": "private",
        "education": "bachelors",
        "marital_status": "married_civ_spouse",
        "occupation": "exec_managerial",
        "relationship": "husband",
        "race": "white",
        "sex": "male",
        "native_country": "united_states",
        "age": 57,
        "fnlgt": 199847,
        "education_num": 13,
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 60
    }
    r = client.post("/inference/", json=payload)
    assert r.json()["Predicted salary"] == ">50k"

