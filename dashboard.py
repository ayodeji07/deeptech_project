# GeoAI Malaria Risk Dashboard
# Author: Ayodeji 
# Purpose: Interactive web app to visualize malaria risk models, maps, and insights
# Dependencies: dash, plotly, geopandas, pandas, libpysal, esda, numpy, scipy, joblib
# Run: python dashboard.py → Open http://127.0.0.1:8050/

import dash  # Web framework for dashboard
from dash import dcc, html, Input, Output  # Core components
import plotly.express as px  # For maps/charts
import plotly.graph_objects as go  # For heatmaps
import geopandas as gpd  # Spatial data handling
import pandas as pd  # Data manipulation
import json  # JSON serialization
from libpysal.weights import KNN  # Spatial weights
from esda.getisord import G_Local  # Hotspot analysis
import numpy as np  # Numerical ops
from scipy.interpolate import griddata  # Interpolation for intensity map
import joblib  # Parallelism control (to avoid MKL crashes)
import dash_bootstrap_components as dbc  # For styling/modal

# Set joblib to serial (n_jobs=1) globally to avoid MKL/parallel crashes
joblib.parallel_config(n_jobs=1, backend='threading')  # Serial mode: Prevents worker crashes

# === DATA LOADING: Subset columns to minimize memory (your GeoJSON has ~95 cols) ===
# Key columns from your sample (DHSID for ID, LAT/LONG for coords, env vars for analysis)
needed_cols = ['DHSID', 'LATNUM', 'LONGNUM', 'Malaria_Prevalence_2020', 'Precipitation_2020', 
               'Enhanced_Vegetation_Index_2020', 'Land_Surface_Temperature_2020',
               'Mean_Temperature_2020', 'ITN_Coverage_2020', 'geometry']

# Load with column filter (fixes deprecation; reduces load time/memory)
full_gdf = gpd.read_file('dashboard_data/merged_gdf.geojson',
                         columns=needed_cols)  # Only load essentials

# Subsample for low-memory testing (25% → ~487 rows; set frac=1.0 for full ~1947 rows)
full_gdf = full_gdf.sample(frac=0.25, random_state=42)  # Random seed for reproducibility

# Load Nigeria boundaries (ADM1 states for basemap)
nigeria_gdf = gpd.read_file('dashboard_data/nga_admin1.geojson')

# === SPATIAL ANALYSIS: Precompute hotspots (Getis-Ord Gi*) ===
# Project to geographic CRS for lat/lon access
full_gdf = full_gdf.to_crs('EPSG:4326')
# KNN weights (k=5 neighbors; faster than Queen for points)
w = KNN.from_dataframe(full_gdf, k=5)
# Gi* on prevalence (permutations=0 for speed; uses asymptotic p-values)
gi = G_Local(full_gdf['Malaria_Prevalence_2020'], w, permutations=0)
full_gdf['gi_star'] = gi.Gs  # Gi* statistic (positive = hotspot)
full_gdf['gi_p'] = gi.p_norm  # p-value (low = significant)

# Precompute intensity grid (even coarser for speed: 0.2 deg ~22km)
bounds = nigeria_gdf.total_bounds
grid_res = 0.2  # Coarser to reduce array size/memory
x_grid = np.arange(bounds[0], bounds[2], grid_res)
y_grid = np.arange(bounds[1], bounds[3], grid_res)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
points = np.array(full_gdf.geometry.apply(lambda g: (g.x, g.y)).tolist())
values = full_gdf['Malaria_Prevalence_2020'].values
# Linear interpolation (fast; cubic is smoother but slower)
interp_grid = griddata(points, values, (x_mesh, y_mesh), method='linear', rescale=True)

# === GEOJSON PREP: Fix serialization issues ===
# Convert any datetime cols to str (avoids JSON error in choropleth if used)
for col in nigeria_gdf.columns:
    if pd.api.types.is_datetime64_any_dtype(nigeria_gdf[col]):
        nigeria_gdf[col] = nigeria_gdf[col].astype(str)
nigeria_geojson = json.loads(nigeria_gdf.to_json())  # For potential choropleth

# === CHART DATA: Hardcoded examples (replace with loads from your 03_modeling outputs) ===
# Feature Importance (from XGB; expand with your actual features)
importances_xgb = pd.Series({
    'Vegetation (Green Areas)': 0.3,  # Relatable name
    'Rainfall Levels': 0.25,
    'Evapotranspiration': 0.15,
    'Elevation': 0.1,
    'NDVI (Plant Health)': 0.08,
    # Add all your features with relatable names
}).sort_values(ascending=False)

# SHAP Summary (mean impacts; from your KernelExplainer)
shap_df = pd.DataFrame({
    'Factor': ['Vegetation (Green Areas)', 'Rainfall Levels', 'Evapotranspiration', 'Bed Net Coverage', 'Land Temperature'],
    'Impact': [0.1, 0.05, -0.03, -0.04, 0.02]  # Positive = increases risk; added more to avoid empty chart
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
            html.P("Clustering Map: Red points show where malaria cases are close together (higher outbreak risk)."),
            html.P("Intensity Map: Yellow to red shows average malaria levels in areas (red = higher)."),
            html.P("Risk Factors: Charts show what makes malaria more or less likely, like rain or bed nets."),
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
            html.P("This heatmap shows how factors like rainfall relate to malaria cases. Red = strong positive link (e.g., more rain = more cases); Blue = negative link (e.g., more bed nets = fewer cases).", style={'textAlign': 'center', 'margin': '10px'}),
            dcc.Graph(id='correlation-heatmap', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Survey Data Table', value='tab-6', children=[
            html.P("Table of survey data points. Scroll to see details like prevalence and location.", style={'textAlign': 'center', 'margin': '10px'}),
            dash.dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in full_gdf.drop(columns='geometry').columns],
                data=full_gdf.drop(columns='geometry').to_dict('records'),  # Drop geometry for serialization
                page_size=20,
                style_table={'overflowX': 'auto', 'width': '100%'},
                style_cell={'textAlign': 'left', 'fontSize': '12px'}
            )
        ])
    ])
])


# === CALLBACKS: Update graphs on tab change ===

# Callback for help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2):
    return True if n1 else False

@app.callback(Output('hotspot-map', 'figure'), Input('tabs', 'value'))
def render_hotspot(tab):
    if tab == 'tab-1':
        # Scatter map: Color by Gi* (hotspots red), size by significance
        fig = px.scatter_mapbox(
            full_gdf,
            lat=full_gdf.geometry.y,  # Extract lat from geometry
            lon=full_gdf.geometry.x,  # Extract lon from geometry
            color='gi_star',
            size='gi_p',  # Smaller = more significant (low p)
            color_continuous_scale='Reds',
            zoom=5,  # Nigeria view
            center={"lat": 9.08, "lon": 7.48},  # Central Nigeria
            mapbox_style="open-street-map",  # Free basemap
            title="Malaria Hotspots (Getis-Ord Gi*)",
            hover_data=['DHSID', 'Malaria_Prevalence_2020']  # Tooltip info
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})  # Tight layout
        return fig
    return go.Figure()  # Empty if not active tab

@app.callback(Output('intensity-map', 'figure'), Input('tabs', 'value'))
def render_intensity(tab):
    if tab == 'tab-2':
        # Flatten interpolated grid to DF for scatter map
        intensity_df = pd.DataFrame({
            'lon': x_mesh.flatten(),
            'lat': y_mesh.flatten(),
            'intensity': interp_grid.flatten()
        }).dropna()  # Remove NaN predictions
        # Scatter map: Color by interpolated prevalence
        fig = px.scatter_mapbox(
            intensity_df,
            lat='lat',
            lon='lon',
            color='intensity',
            color_continuous_scale='YlOrRd',  # Yellow-low, red-high
            zoom=5,
            center={"lat": 9.08, "lon": 7.48},
            mapbox_style="open-street-map",
            title="Malaria Risk Intensity (Linear Interpolation)",
            opacity=0.6  # Semi-transparent for density effect
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    return go.Figure()

@app.callback(Output('importance-chart', 'figure'), Input('tabs', 'value'))
def render_importance(tab):
    if tab == 'tab-3':
        # Horizontal bar for readability
        fig = px.bar(importances_xgb, orientation='h', title="XGBoost Feature Importance")
        fig.update_layout(xaxis_title="Importance", yaxis_title="Features")
        return fig
    return go.Figure()

@app.callback(Output('shap-summary', 'figure'), Input('tabs', 'value'))
def render_shap(tab):
    if tab == 'tab-4':
        # Horizontal bar (positive right, negative left)
        fig = px.bar(shap_df, x='SHAP', y='Feature', orientation='h',
                     title="SHAP Summary (Impact on Model Output)",
                     color='SHAP', color_continuous_scale='RdBu_r')  # Red positive, blue negative
        fig.update_layout(xaxis_title="Mean SHAP Value", yaxis_title="Features")
        return fig
    return go.Figure()

@app.callback(Output('correlation-heatmap', 'figure'), Input('tabs', 'value'))
def render_corr(tab):
    if tab == 'tab-5':
        # Heatmap with values overlaid
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',  # Red positive, blue negative
            text=corr.round(2),  # Show values
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig.update_layout(title="Spearman Correlation Heatmap (Env Factors vs. Prevalence)")
        return fig
    return go.Figure()

server = app.server   # Required for gunicorn/Render deployment

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)