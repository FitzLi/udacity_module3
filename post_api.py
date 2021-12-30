import requests
import json

url = 'https://udacity-module3.herokuapp.com/inference/'
data = {
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
response = requests.post(url, json.dumps(data))
print(f'POST status code is: {response.status_code}')
print(f'POST result of model inference is: {response.json()}')
