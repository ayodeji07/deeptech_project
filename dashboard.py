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

# Set joblib to serial (n_jobs=1) globally to avoid MKL/parallel crashes
joblib.parallel_config(n_jobs=1, backend='threading')

# Load data with column selection to reduce memory (only needed columns)
needed_cols = ['DHSID', 'LATNUM', 'LONGNUM', 'Malaria_Prevalence_2020', 'Precipitation_2020', 
               'Enhanced_Vegetation_Index_2020', 'Land_Surface_Temperature_2020',
               'Mean_Temperature_2020', 'ITN_Coverage_2020', 'geometry']  

full_gdf = gpd.read_file(r'C:\Users\Hp\Documents\capstone_project\notebooks\data\processed\merged_gdf.geojson',
                         columns=needed_cols)  

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
    'Enhanced_Vegetation_Index_2020': 0.3,
    'Precipitation_2020': 0.25,
    'PET_2020': 0.15,
    'elevation': 0.1,
    'ndvi': 0.08,
    # Add all your features
}).sort_values(ascending=False)

# SHAP Summary (example bar data)
shap_df = pd.DataFrame({
    'Feature': ['Enhanced_Vegetation_Index_2020', 'Precipitation_2020', 'PET_2020'],
    'SHAP': [0.1, 0.05, -0.03]  # Positive/negative impacts
})

# Spearman Correlation (use actual columns from your GeoJSON sample)
selected_cols = ['Precipitation_2020', 'Enhanced_Vegetation_Index_2020', 'Land_Surface_Temperature_2020',
                 'Mean_Temperature_2020', 'ITN_Coverage_2020', 'Malaria_Prevalence_2020']  # Adjusted to existing
corr = full_gdf[selected_cols].corr(method='spearman')

# App setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("GeoAI Malaria Risk Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Hotspot Map', value='tab-1', children=[
            dcc.Graph(id='hotspot-map', style={'height': '80vh'})
        ]),
        dcc.Tab(label='Intensity Map', value='tab-2', children=[
            dcc.Graph(id='intensity-map', style={'height': '80vh'})
        ]),
        dcc.Tab(label='Feature Importance', value='tab-3', children=[
            dcc.Graph(id='importance-chart', style={'height': '60vh'})
        ]),
        dcc.Tab(label='SHAP Summary', value='tab-4', children=[
            dcc.Graph(id='shap-summary', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Spearman Correlation', value='tab-5', children=[
            dcc.Graph(id='correlation-heatmap', style={'height': '60vh'})
        ]),
        dcc.Tab(label='Data Table', value='tab-6', children=[
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

# Callbacks for dynamic updates (optional; static for simplicity, but add inputs if needed)
@app.callback(Output('hotspot-map', 'figure'), Input('tabs', 'value'))
def render_hotspot(tab):
    if tab == 'tab-1':
        fig = px.scatter_mapbox(
            full_gdf,
            lat=full_gdf.geometry.y,
            lon=full_gdf.geometry.x,
            color='gi_star',
            size='gi_p',  # Size by significance
            color_continuous_scale='Reds',
            zoom=5,
            center={"lat": 9.08, "lon": 7.48},
            mapbox_style="open-street-map",
            title="Malaria Hotspots (Gi*)"
        )
        return fig
    return {}

@app.callback(Output('intensity-map', 'figure'), Input('tabs', 'value'))
def render_intensity(tab):
    if tab == 'tab-2':
        # Flatten grid for scatter (faster alternative to imshow in Dash)
        intensity_df = pd.DataFrame({
            'lon': x_mesh.flatten(),
            'lat': y_mesh.flatten(),
            'intensity': interp_grid.flatten()
        }).dropna()  # Drop NaNs
        fig = px.scatter_mapbox(
            intensity_df,
            lat='lat',
            lon='lon',
            color='intensity',
            color_continuous_scale='YlOrRd',
            zoom=5,
            center={"lat": 9.08, "lon": 7.48},
            mapbox_style="open-street-map",
            title="Malaria Risk Intensity (Interpolated)"
        )
        return fig
    return {}

@app.callback(Output('importance-chart', 'figure'), Input('tabs', 'value'))
def render_importance(tab):
    if tab == 'tab-3':
        fig = px.bar(importances_xgb, title="XGBoost Feature Importance")
        fig.update_layout(xaxis_title="Features", yaxis_title="Importance")
        return fig
    return {}

@app.callback(Output('shap-summary', 'figure'), Input('tabs', 'value'))
def render_shap(tab):
    if tab == 'tab-4':
        fig = px.bar(shap_df, x='SHAP', y='Feature', title="SHAP Summary (Impact on Model Output)")
        fig.update_layout(xaxis_title="Mean SHAP Value", yaxis_title="Features")
        return fig
    return {}

@app.callback(Output('correlation-heatmap', 'figure'), Input('tabs', 'value'))
def render_corr(tab):
    if tab == 'tab-5':
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            texttemplate="%{z:.2f}"
        ))
        fig.update_layout(title="Spearman Correlation Heatmap")
        return fig
    return {}

server = app.server  # For deployment

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)