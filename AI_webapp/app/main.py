from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.models.models import ModeLInput, ModeLOutput
import pickle
from pathlib import Path
import pandas as pd

models = {}
pkl_asset_filename = Path("app/pickle_assetModel.pkl")
pkl_liability_filename = Path("app/pickle_liabilityModel.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    with open(pkl_asset_filename, 'rb') as file:
        models['assetModel'] = pickle.load(file)

    with open(pkl_liability_filename, 'rb') as file:
        models['liabilityModel'] = pickle.load(file)

    yield
    # Clean up the ML models and release the resources
    models.clear()


app = FastAPI(title='Banking System Capital Prediction App', lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict",  response_model=ModeLOutput)
async def predict(params: ModeLInput):
    rates = params.interest_rate
    year = params.year
    inputData = pd.DataFrame(
        {'Year': year, 'Interest Rates': rates})
    asset = models["assetModel"].predict(inputData)
    liability = models["liabilityModel"].predict(inputData)
    capitals = [a - b for a, b in zip(asset.tolist(), liability.tolist())]
    print('result', capitals)
    return {"result": capitals}
