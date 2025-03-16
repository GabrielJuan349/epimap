import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import datetime
import requests
import json
import geopandas as gpd
from io import BytesIO
from collections import defaultdict
# Eliminadas las importaciones de folium
import zipfile
import os
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="EpiMap - Sistema de Alerta Temprana",
    page_icon="🦠",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .alert-box {
        background-color: #ffecb3;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .stAlert > div {
        padding-top: 15px;
        padding-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cabecera de la aplicación
st.markdown('<h1 class="main-header">EpiMap: Sistema de Alerta Temprana para Enfermedades Emergentes</h1>', unsafe_allow_html=True)

# Descripción del proyecto
st.markdown("""
EpiMap combina análisis de datos en tiempo real con mapeo interactivo para proporcionar vigilancia epidemiológica avanzada. 
La plataforma rastrea brotes, monitorea la propagación de enfermedades y genera alertas que permiten a gobiernos, 
organizaciones de salud y al público responder de manera proactiva ante amenazas sanitarias emergentes.
""")

# Obtener la ruta del archivo CSV usando una ruta absoluta basada en la ubicación del script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
country_path = os.path.join(project_root, 'data', 'country_codes.csv')
epi_path = os.path.join(project_root, 'data', 'epi_codes.csv')

# Cargar el dataframe con la ruta absoluta
options_dataframe = pd.read_csv(country_path)
options_countries = options_dataframe.iloc[:, 1].unique().tolist()
del options_dataframe

# Cargar el dataframe con la ruta absoluta
options_dataframe = pd.read_csv(epi_path)
options_epi = options_dataframe.iloc[:, 2].unique().tolist()
del options_dataframe

# Función para obtener datos de la API
@st.cache_data(ttl=3600)  # Caché de 1 hora
def get_api_data(endpoint):
    """Obtiene datos de la API especificada"""
    return get_sample_data(endpoint)

def get_data(data_type):
    """Obtiene datos de la API especificada"""
    endpoints = {
        1 : "map_data",
        2 : "notifications",
        3: "age_data",
        4: "top5_illness"
    }
    try:
        base_url = "https://localhost:8000/"
        response = requests.get(f"{base_url}/{endpoints[data_type]}")
        response.raise_for_status()
        return transform_data(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener datos de la API: {e}")
        # return get_sample_data(endpoint)

def transform_data(data):
    """"Transforma los datos de un json a un DataFrame"""
    return pd.read_json(data)
    
# Función para obtener datos de muestra
def get_sample_data(endpoint):
    np.random.seed(42)
    current_date = datetime.datetime.now()
    
    if endpoint == "outbreaks":
        countries = {
            "Brazil": [-15.77972, -47.92972],
            "Mexico": [19.42847, -99.12766],
            "Colombia": [4.60971, -74.08175],
            "Argentina": [-34.61315, -58.37723],
            "Peru": [-12.04318, -77.02824]
        }
        
        map_data = []
        for country, coords in countries.items():
            intensity = np.random.randint(1, 10)
            points = np.random.randn(intensity * 5, 2) / [20, 20] + coords
            for point in points:
                disease = np.random.choice(["COVID-19", "Meningitis", "Influenza", "Malaria", "Cholera"])
                severity = np.random.choice(["Bajo", "Medio", "Alto"], p=[0.7, 0.2, 0.1])
                days_ago = np.random.randint(0, 30)
                report_date = current_date - datetime.timedelta(days=days_ago)
                
                map_data.append({
                    "lat": float(point[0]),
                    "lon": float(point[1]),
                    "country": country,
                    "disease": disease,
                    "severity": severity,
                    "report_date": report_date.strftime("%Y-%m-%d"),
                    "days_since_report": days_ago,
                    "cases": np.random.randint(1, 100),
                    "risk_level": np.random.randint(1, 5)
                })
        return {"data": map_data}
    
    elif endpoint == "alerts":
        return {
            "data": [
                {"disease": "Meningitis", "location": "Brazil - São Paulo", "level": "Alto", "date": "2025-03-14"},
                {"disease": "Influenza", "location": "México - Ciudad de México", "level": "Medio", "date": "2025-03-13"},
                {"disease": "COVID-19", "location": "Argentina - Buenos Aires", "level": "Bajo", "date": "2025-03-12"}
            ]
        }
    
    elif endpoint == "trends":
        diseases = ["COVID-19", "Meningitis", "Influenza", "Malaria", "Cholera"]
        trends_data = []
        
        for disease in diseases:
            base = np.random.randint(10, 100)
            for i in range(30):
                date = (current_date - datetime.timedelta(days=29-i)).strftime("%Y-%m-%d")
                trend = int(np.random.normal() * 10 + base)
                trends_data.append({
                    "date": date,
                    "disease": disease,
                    "cases": trend
                })
        return {"data": trends_data}
    
    return {"data": []}

# Función para cargar datos geoespaciales de países
@st.cache_data
def load_country_shapes():
    """Carga las formas geoespaciales de los países de América Latina directamente desde Natural Earth"""
    try:
        # URL actualizada a una fuente funcional (CDN de NACIS)
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        
        # Descargar el archivo zip
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Error al descargar datos: HTTP {response.status_code}")
            return None
            
        # Crear un directorio temporal
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Guardar y extraer el zip
            zip_path = os.path.join(tmpdirname, "countries.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extraer el archivo zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            # Buscar el archivo shapefile
            shp_files = [f for f in os.listdir(tmpdirname) if f.endswith('.shp')]
            if not shp_files:
                st.warning("No se encontró ningún archivo shapefile en el zip")
                return None
                
            # Cargar el primer shapefile encontrado
            world = gpd.read_file(os.path.join(tmpdirname, shp_files[0]))
            
            # Filtrar solo países de América Latina
            countries = world[world['ADMIN'].isin(options_countries)]
            
            if countries.empty:
                st.warning("No se encontraron países latinoamericanos en los datos")
                return json.loads(world.to_json())
            
            return json.loads(countries.to_json())
    except Exception as e:
        st.error(f"Error al cargar datos geoespaciales: {str(e)}")
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return json.loads(world.to_json())
        except Exception as e2:
            st.error(f"Error en fallback: {str(e2)}")
            return None


# Funciones para manejar colores según severidad y riesgo
def get_color_for_severity(severity):
    if severity == "Alto":
        return [255, 0, 0, 160]
    elif severity == "Medio":
        return [255, 165, 0, 160]
    else:
        return [255, 255, 0, 160]

def get_color_for_risk_level(risk_level):
    if risk_level >= 5:
        return [128, 0, 0, 200]
    elif risk_level == 4:
        return [255, 0, 0, 200]
    elif risk_level == 3:
        return [255, 140, 0, 200]
    elif risk_level == 2:
        return [255, 255, 0, 200]
    else:
        return [0, 128, 0, 200]

def get_severity_value(severity):
    if severity == "Alto":
        return 3
    elif severity == "Medio":
        return 2
    else:
        return 1

# Función para obtener GeoJSON de un país
def get_country_geojson(country_name):
    """Obtiene el GeoJSON de un país específico"""
    try:
        if country_shapes and country_shapes.get('features'):
            
            for feature in country_shapes['features']:
                name_fields = ['name', 'NAME', 'NAME_EN', 'ADMIN']
                for field in name_fields:
                    if field in feature['properties'] and feature['properties'][field].lower() == country_name.lower():
                        # Asegurarse de tener una propiedad 'name' consistente para los tooltips
                        feature['properties']['name'] = feature['properties'].get('ADMIN', feature['properties'][field])
                        return feature
            
            st.warning(f"No se encontró el país: {country_name}")
        else:
            st.warning("No hay datos de países cargados")
            
        return None
    except Exception as e:
        st.error(f"Error al obtener datos GeoJSON: {str(e)}")
        return None

# Función create_folium_map eliminada

# Cargar datos de la API
outbreak_data = get_api_data("outbreaks")
alerts_data = get_api_data("alerts")
trends_data = get_api_data("trends")
map_df = pd.DataFrame(outbreak_data["data"])
country_shapes = load_country_shapes()

# Layout de dos columnas
col1, col2 = st.columns([2, 1])



with col1:
    st.markdown('<p class="sub-header">Mapa de Vigilancia Epidemiológica</p>', unsafe_allow_html=True)
    
    # Selector de tipo de mapa eliminado
    
    with st.expander("Opciones de Filtrado", expanded=True):
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            selected_diseases = st.multiselect(
                "Seleccionar enfermedades",
                options=options_epi,
                default=map_df["disease"].unique()
            )
        
        with col_filter2:
            selected_countries = st.multiselect(
                "Seleccionar países",
                options=options_countries,
                default=map_df["country"].unique()
            )
        
        with col_filter3:
            days_range = st.slider(
                "Edad",
                min_value=0,
                max_value=100,
                value=(0, 100)
            )
    
    filtered_df = map_df[
        (map_df["disease"].isin(selected_diseases)) &
        (map_df["country"].isin(selected_countries)) &
        (map_df["days_since_report"] >= days_range[0]) &
        (map_df["days_since_report"] <= days_range[1])
    ]
    
    filtered_df["color"] = filtered_df["severity"].apply(get_color_for_severity)
    
    country_risk = defaultdict(int)
    country_severity = defaultdict(int)
    
    for _, row in filtered_df.iterrows():
        country = row["country"]
        risk = row["risk_level"]
        severity = get_severity_value(row["severity"])
        
        if risk > country_risk[country]:
            country_risk[country] = risk
        
        if severity > country_severity[country]:
            country_severity[country] = severity
    
    view_state = pdk.ViewState(
        latitude=0,
        longitude=-70,
        zoom=3,
        pitch=0
    )
    
    country_layers = []
    
    if country_shapes and country_shapes.get('features'):
        
        for feature in country_shapes['features']:
            country_name = feature['properties'].get('name', feature['properties'].get('ADMIN', ''))
            
            if country_name in selected_countries and country_name in country_risk:
                risk_level = country_risk[country_name]
                color = get_color_for_risk_level(risk_level)
                border_color = color
                fill_color = [color[0], color[1], color[2], 180]
                
                # Añadir el nivel de riesgo y asegurarnos de que el nombre está en español
                feature['properties']['risk_level'] = int(risk_level)
                # Establecer el nombre en español explícitamente
                feature['properties']['name'] = country_name
                
                country_feature = {
                    'type': 'FeatureCollection',
                    'features': [feature]
                }
                
                country_layer = pdk.Layer(
                    'GeoJsonLayer',
                    country_feature,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    wireframe=True,
                    get_fill_color=fill_color,
                    get_line_color=border_color,
                    get_line_width=5,
                    pickable=True,
                    auto_highlight=True
                )
                
                country_layers.append(country_layer)
    
    if country_shapes:
        base_layer = pdk.Layer(
            'GeoJsonLayer',
            country_shapes,
            opacity=0.2,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_fill_color=[135, 206, 250, 50],
            get_line_color=[0, 0, 139, 100],
            get_line_width=1,
            pickable=True
        )
    else:
        base_layer = None
    
    layers = []
    if base_layer:
        layers.append(base_layer)
    layers.extend(country_layers)
    
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "text": "País: {name}\nNivel de riesgo: {risk_level}/5"
        }
    )
    
    st.pydeck_chart(deck)
    
    st.info("Los países están coloreados según el nivel de riesgo más alto detectado en cada uno.")
    
    st.markdown("### Leyenda de Niveles de Riesgo")
    legend_cols = st.columns(5)
    
    risk_descriptions = {
        1: "Muy bajo",
        2: "Bajo",
        3: "Medio",
        4: "Alto",
        5: "Muy alto"
    }
    
    for i, col in enumerate(legend_cols):
        risk_level = i + 1
        color = get_color_for_risk_level(risk_level)
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        col.markdown(f"""
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: {hex_color}; margin-right: 8px;"></div>
                <span>{risk_descriptions[risk_level]}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"**Mostrando {len(filtered_df)} brotes de {len(selected_diseases)} enfermedades en {len(selected_countries)} países**")

with col2:
    st.markdown('<p class="sub-header">Panel de Control</p>', unsafe_allow_html=True)
    
    st.markdown("### 🚨 Alertas Recientes")
    
    for alert in alerts_data["data"]:
        level_color = "red" if alert["level"] == "Alto" else "orange" if alert["level"] == "Medio" else "green"
        st.markdown(
            f"""<div class="alert-box">
                <strong style="color:{level_color}">{alert["level"]}</strong>: {alert["disease"]} en {alert["location"]}
                <br><small>{alert["date"]}</small>
                </div>""", 
            unsafe_allow_html=True
        )
    
    st.markdown("### 📊 Resumen Estadístico")
    
    disease_counts = filtered_df["disease"].value_counts()
    
    fig = px.bar(
        x=disease_counts.index,
        y=disease_counts.values,
        labels={"x": "Enfermedad", "y": "Número de Brotes"},
        color=disease_counts.values,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Eliminar la línea problemática con age_counts
    # age_counts:pd.DataFrame = get_data(3)
    age_counts = pd.DataFrame({
        "age": ["0-4", "5-14", "15-24", "35-45", "45-54", "55-64", "65-74", "75-84", "85-94", "95+"],
        "probability": [0.03, 0.01, 0.02, 0.04, 0.07, 0.12, 0.18, 0.25, 0.35, 0.50]
    })
    
    age_fig = px.bar(
        x=age_counts["age"],
        y=age_counts["probability"],
        labels={"x": "Edad", "y": "Probabilidad de muerte"},
        color=age_counts["probability"],
        color_continuous_scale="Turbo"
    )
    age_fig.update_layout(height=300)
    st.plotly_chart(age_fig, use_container_width=True)
    
    severity_counts = filtered_df["severity"].value_counts()
    fig_pie = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Distribución por Severidad",
        color=severity_counts.index,
        color_discrete_map={"Alto": "red", "Medio": "orange", "Bajo": "yellow"}
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("### 🌎 Países con Mayor Riesgo")
    
    risk_data = pd.DataFrame({
        "País": list(country_risk.keys()),
        "Nivel de Riesgo": list(country_risk.values())
    })
