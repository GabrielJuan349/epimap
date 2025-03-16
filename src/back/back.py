from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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

# Endpoint GET
@app.get("/api/hello")
async def hello_world():
    """Endpoint simple que devuelve un mensaje de saludo"""
    return {"message": "¡Hola desde EpiMap API!"}

# Endpoint POST
@app.post("/notifications")
async def process_data(item: Item):
    """Endpoint que procesa datos recibidos en formato JSON"""
    result = {
        "processed_name": item.name.upper(),
        "processed_value": item.value * 2,
        "status": "success"
    }
    return result

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
async def age_data(item: Item):
    """Endpoint que procesa datos recibidos en formato JSON"""
    result = {
        "processed_name": item.name.upper(),
        "processed_value": item.value * 2,
        "status": "success"
    }
    return result

# Top 5 illnes endpoint
@app.post("/top5_illness")
async def illness_data():
    

# Para ejecutar la aplicación directamente
if __name__ == "__main__":
    uvicorn.run("back:app", host="0.0.0.0", port=8000, reload=True)
