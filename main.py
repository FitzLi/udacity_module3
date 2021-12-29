import os
import pathlib
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

#if "DYNO" in os.environ and os.path.isdir(".dvc"):
#    os.system("dvc config core.no_scm true")
#    if os.system(f"dvc pull") != 0:
#        exit("dvc pull failed")
#    os.system("rm -r .dvc .apt/usr/lib/dvc")
#os.system('pip install "dvc[s3]"')
os.system('git init')
os.system('dvc init model')
os.system('dvc pull')


app = FastAPI()

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

class Feature(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
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
        }


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'model.pkl' 
ENCODER_PATH = MODEL_DIR / 'encoder.pkl'
LB_PATH = MODEL_DIR / 'lb.pkl'

MODEL = None
ENCODER = None 
LB = None

@app.on_event("startup")
def models_loading():
    global MODEL
    global ENCODER
    global LB
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as file_in:
            MODEL = pickle.load(file_in)
    if ENCODER_PATH.exists():
        with open(ENCODER_PATH, 'rb') as file_in:
            ENCODER = pickle.load(file_in)
    if LB_PATH.exists():
        with open(LB_PATH, 'rb') as file_in:
            LB = pickle.load(file_in)
  
@app.get("/")
def read_index():
    return "What's up, man~~~!"

@app.post("/inference/")
async def predict_salary(input_features: Feature):
    input_num = [getattr(input_features, col) for col in NUM_FEATURES]
    input_cat = [[getattr(input_features, col) for col in CAT_FEATURES]]
    input_cat = ENCODER.transform(input_cat).tolist()[0]
    processed_features = input_num + input_cat
    pred = MODEL.predict([processed_features])
    pred = LB.inverse_transform(pred)[0]
    return {"Predicted salary": pred}
