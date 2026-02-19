import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd
import json
from libpysal.weights import KNN
from esda.getisord import G_Local
import numpy as np
from scipy.interpolate import griddata
import joblib  # To control parallelism
import dash_bootstrap_components as dbc  # For styling/modal

# Set joblib to serial (n_jobs=1) globally to avoid MKL/parallel crashes
joblib.parallel_config(n_jobs=1, backend='threading')

# Load data with column selection to reduce memory (only needed columns)
needed_cols = ['DHSID', 'LATNUM', 'LONGNUM', 'Malaria_Prevalence_2020', 'Precipitation_2020', 
               'Enhanced_Vegetation_Index_2020', 'Land_Surface_Temperature_2020',
               'Mean_Temperature_2020', 'ITN_Coverage_2020', 'geometry']  # Adjust to your key features

full_gdf = gpd.read_file(r'C:\Users\Hp\Documents\capstone_project\notebooks\data\processed\merged_gdf.geojson',
                         columns=needed_cols)  # Updated from 'include_fields' (deprecated)

# Subsample to ~500 rows for low-memory machines (remove or increase frac for full data)
full_gdf = full_gdf.sample(frac=0.25, random_state=42)  # ~487 rows → faster computations

nigeria_gdf = gpd.read_file(r'C:\Users\Hp\Documents\capstone_project\data\raw\nga_admin_boundaries.geojson\nga_admin1.geojson')

# Precompute hotspots (using KNN for speed; permutations=0 to skip sim for speed/no p-values)
full_gdf = full_gdf.to_crs('EPSG:4326')  # Ensure geographic CRS
w = KNN.from_dataframe(full_gdf, k=5)
gi = G_Local(full_gdf['Malaria_Prevalence_2020'], w, permutations=0)  # No sim → faster, but no p_sim
full_gdf['gi_star'] = gi.Gs
full_gdf['gi_p'] = gi.p_norm  # Use asymptotic p-values instead

# Precompute intensity grid (even coarser for speed: 0.2 deg ~22km)
bounds = nigeria_gdf.total_bounds
grid_res = 0.2  # Coarser to reduce array size/memory
x_grid = np.arange(bounds[0], bounds[2], grid_res)
y_grid = np.arange(bounds[1], bounds[3], grid_res)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
points = np.array(full_gdf.geometry.apply(lambda g: (g.x, g.y)).tolist())
values = full_gdf['Malaria_Prevalence_2020'].values
interp_grid = griddata(points, values, (x_mesh, y_mesh), method='linear', rescale=True)

# Prepare GeoJSON for choropleth (convert Timestamps to strings to avoid JSON error)
for col in nigeria_gdf.columns:
    if pd.api.types.is_datetime64_any_dtype(nigeria_gdf[col]):
        nigeria_gdf[col] = nigeria_gdf[col].astype(str)  # Convert Timestamp to str
nigeria_geojson = json.loads(nigeria_gdf.to_json())

# Example data for charts (replace with your actual from 03_modeling outputs)
# Feature Importance (load or hardcode from your bars)
importances_xgb = pd.Series({
    'Vegetation (Green Areas)': 0.3,  # Relatable name
    'Rainfall Levels': 0.25,
    'Evapotranspiration': 0.15,
    'Elevation': 0.1,
    'NDVI (Plant Health)': 0.08,
    # Add all your features with relatable names
}).sort_values(ascending=False)

# SHAP Summary (example bar data)
shap_df = pd.DataFrame({
    'Factor': ['Vegetation (Green Areas)', 'Rainfall Levels', 'Evapotranspiration'],
    'Impact': [0.1, 0.05, -0.03]  # Positive = increases risk
})

# Spearman Correlation (on selected env vars)
selected_cols = ['Precipitation_2020', 'Enhanced_Vegetation_Index_2020', 'Land_Surface_Temperature_2020',
                 'Mean_Temperature_2020', 'ITN_Coverage_2020', 'Malaria_Prevalence_2020']
corr = full_gdf[selected_cols].corr(method='spearman')
corr.columns = ['Rainfall', 'Vegetation', 'Land Temp', 'Mean Temp', 'Bed Net Coverage', 'Malaria Cases']  # Relatable labels
corr.index = corr.columns

# App setup with Bootstrap for clean styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Modal for help/explanations (pops up on button click)
help_modal = dbc.Modal(
    [
        dbc.ModalHeader("Dashboard Guide"),
        dbc.ModalBody([
            html.P("This dashboard helps understand malaria risks in Nigeria using simple maps and charts."),
            html.P("Hotspots: Red areas show where malaria cases are clustered (higher chance of outbreaks)."),
            html.P("Intensity: Yellow to red map shows average malaria levels across regions."),
            html.P("Factors: Charts explain what influences malaria, like rain or green areas."),
            html.P("Click tabs to explore! For questions, contact ayodeji@email.com.")
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ml-auto")),
    ],
    id="help-modal",
    is_open=False,
)

app.layout = html.Div([
    html.H1("Nigeria Malaria Risk Explorer", style={'textAlign': 'center', 'margin': '20px'}),
    html.Div([
        html.P("Welcome! This tool shows malaria patterns in Nigeria. Use tabs to view maps and charts. Click '?' for help.", style={'textAlign': 'center'}),
        dbc.Button("?", id="open-modal", className="mb-3"),
        help_modal  # Add modal to layout
    ]),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Malaria Clustering Map', value='tab-1', children=[
            html.P("This map shows areas where malaria cases are grouped together. Red points mean higher clustering and risk based on health surveys.", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='hotspot-map', style={'height': '80vh'})
        ]),
        dcc.Tab(label='Malaria Intensity Map', value='tab-2', children=[
            html.P("This map shows average malaria levels across Nigeria. Yellow = low, red = high, estimated from survey data.", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='intensity-map', style={'height': '80vh'})
        ]),
        dcc.Tab(label='Key Risk Factors', value='tab-3', children=[
            html.P("This chart shows what influences malaria risk the most, like rainfall or green areas.", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='importance-chart', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Factor Impacts', value='tab-4', children=[
            html.P("This chart explains how each factor affects malaria predictions. Positive = increases risk, negative = decreases.", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='shap-summary', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Factor Relationships', value='tab-5', children=[
            html.P("This heatmap shows how factors like rainfall relate to malaria cases. Red = strong positive link.", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='correlation-heatmap', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Survey Data Table', value='tab-6', children=[
            html.P("Table of survey data points. Scroll to see details like prevalence and location.", style={'textAlign': 'center', 'margin': '10px'}),
            dash.dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in full_gdf.drop(columns='geometry').columns],
                data=full_gdf.drop(columns='geometry').to_dict('records'),  # Drop geometry for serialization
                page_size=20,
                style_table={'overflowX': 'auto'}
            )
        ])
    ])
])

# Callback for help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2):
    return True if n1 else False

# ... (keep all your existing callbacks from before — no changes needed here)

server = app.server   # Required for gunicorn/Render deployment

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)