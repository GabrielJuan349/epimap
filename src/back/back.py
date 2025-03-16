from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.back.data_modification import modify_dataset_country, modify_dataset_epi
from src.back.perplexity_req import PerplexityAPI
from src.models.model import get_prediction
import pandas as pd
import json
import os

# Creamos la aplicación FastAPI
app = FastAPI(title="EpiMap API")



# Configuración de CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para los datos del POST
class Item(BaseModel):
    name: str
    value: float
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
del current_dir
train_path=os.path.join(project_root, 'data', 'train.csv')
del project_root
train_data = pd.read_csv(train_path)
del train_path

data = get_prediction(train_data, ['Deaths2-6', 'Deaths7-8', 'Deaths9-10', 'Deaths11-12', 'Deaths13-14', 'Deaths15-16', 'Deaths17-18', 'Deaths19-20', 'Deaths21-22', 'Deaths23-24', 'Deaths25-26'])
data = modify_dataset_country(data,False)
data = modify_dataset_epi(data)
del train_data

pa = PerplexityAPI()
# Endpoint GET
@app.get("/api/hello")
async def hello_world():
    """Endpoint simple que devuelve un mensaje de saludo"""
    return {"message": "¡Hola desde EpiMap API!"}

# Endpoint POST
@app.post("/notifications")
async def process_data():
    """Endpoint que procesa datos recibidos en formato JSON"""
    response = pa.ask(data.to_json(orient='records'))
    return response

# map data endpoint
@app.post("/map_data")
async def process_data(item: Item):
    """Endpoint que procesa datos recibidos en formato JSON"""
    result = {
        "processed_name": item.name.upper(),
        "processed_value": item.value * 2,
        "status": "success"
    }
    return result

# Age data endpoint
@app.post("/age_data")
async def age_data():
    """Endpoint que procesa datos recibidos en formato JSON"""
    result = data.groupby('Cause_Code').sum().sort_values(by='Deaths', ascending=False)
    return result

# Top 5 illnes endpoint
@app.post("/top5_illness")
async def illness_data(countries: list):
    """"Endpoint that sends the top 5 illness from the result of the model"""
    # Filter data to only include countries from the input list
    data_ill = data[data['country'].isin(countries)]
    data_ill = data_ill.groupby('Cause_Code').sum().sort_values(by='Deaths', ascending=False).head(5)
    
    # Prepare the result to return
    result = data_ill.to_dict(orient='records')
    return result

# Para ejecutar la aplicación directamente
if __name__ == "__main__":
    uvicorn.run("back:app", host="0.0.0.0", port=8000, reload=True)
