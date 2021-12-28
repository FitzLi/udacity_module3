import pathlib
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

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


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'model.pkl' 
ENCODER_PATH = MODEL_DIR / 'encoder.pkl'
LB_PATH = MODEL_DIR / 'lb.pkl'

MODEL = None
ENCODER = None 
LB = None

@app.on_event("startup")
def on_startup():
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
    return f"What's up, man~~~ {dir(MODEL)}"

@app.post("/inference/")
async def predict_salary(input_features: Feature):
    input_num = [getattr(input_features, col) for col in NUM_FEATURES]
    input_cat = [[getattr(input_features, col) for col in CAT_FEATURES]]
    input_cat = ENCODER.transform(input_cat).tolist()[0]
    processed_features = input_num + input_cat
    pred = MODEL.predict([processed_features])
    pred = LB.inverse_transform(pred)[0]
    return {"Predicted salary": pred}
