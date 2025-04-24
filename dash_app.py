# 7. dash_app.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import dash
import base64
import io
import json
import flask

app = Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])
app.title = 'Customer Segmentation (RFM)'

server = app.server

# Global state
store = {
    "df": None,
    "model": None,
    "scaler": None,
    "best_model_name": None,
    "model_scores_df": None
}

def compute_rfm(df):
    df = df[df['Quantity'] > 0]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    latest_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def clean_and_preprocess(df):
    rfm = compute_rfm(df)
    Q1 = rfm[['Recency', 'Frequency', 'Monetary']].quantile(0.25)
    Q3 = rfm[['Recency', 'Frequency', 'Monetary']].quantile(0.75)
    IQR = Q3 - Q1
    rfm = rfm[~((rfm < (Q1 - 1.5 * IQR)) | (rfm > (Q3 + 1.5 * IQR))).any(axis=1)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm, X_scaled, scaler

def evaluate_models(X_scaled):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)

    dbscan = DBSCAN(eps=0.9, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

    model_scores_df = pd.DataFrame({
        'Model': ['KMeans', 'DBSCAN'],
        'Silhouette Score': [kmeans_score, dbscan_score]
    })

    best_model = kmeans if kmeans_score >= dbscan_score else dbscan
    best_labels = kmeans_labels if kmeans_score >= dbscan_score else dbscan_labels
    best_score = kmeans_score if kmeans_score >= dbscan_score else dbscan_score
    best_name = "KMeans" if kmeans_score >= dbscan_score else "DBSCAN"

    return best_model, best_labels, best_score, best_name, model_scores_df

def get_chart_figure(chart_type):
    df = store['df']
    if df is None or 'Cluster' not in df.columns:
        return {}

    if chart_type == 'scatter':
        return px.scatter(df, x='Recency', y='Monetary', color='Cluster', hover_data=['Frequency'], title='Customer Clusters (RFM)')
    elif chart_type == 'pie':
        return px.pie(df, names='Cluster', title='Distribution of Customers by Cluster')
    elif chart_type == 'bar':
        return px.bar(df.groupby('Cluster')['Monetary'].sum().reset_index(), x='Cluster', y='Monetary', title='Total Monetary Value per Cluster')
    elif chart_type == 'freq_rec':
        return px.scatter(df, x='Frequency', y='Recency', color='Cluster', title='Frequency vs Recency by Cluster')
    elif chart_type == 'comparison' and store['model_scores_df'] is not None:
        return px.bar(store['model_scores_df'], x='Model', y='Silhouette Score', title='Model Performance Comparison', text='Silhouette Score')
    return {}

app.layout = html.Div([
    dcc.Download(id="download-table"),
    dcc.Download(id="download-image"),

    html.Div([
        html.H1("Customer Segmentation Dashboard", className="text-center text-primary mb-4"),

        html.Div([
            html.H3("1. Upload Transactions CSV", className="text-secondary"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'marginBottom': '20px'
                },
                multiple=False
            ),
            html.Div(id='file-upload-output', className="mb-4"),
        ]),

        html.H3("2. Cluster Visualizations", className="text-secondary"),
        dcc.Tabs(id='charts-tabs', value='scatter', children=[
            dcc.Tab(label='RFM Scatter Plot', value='scatter'),
            dcc.Tab(label='Cluster Distribution Pie', value='pie'),
            dcc.Tab(label='Monetary by Cluster', value='bar'),
            dcc.Tab(label='Frequency vs Recency', value='freq_rec'),
            dcc.Tab(label='Model Comparison', value='comparison')
        ]),
        html.Div([
            dcc.Graph(id='cluster-plot'),
            html.Button('Download Chart', id='download-btn', className="btn btn-outline-secondary mt-2 me-2"),
            html.Button('Download RFM Table', id='download-table-btn', className="btn btn-outline-success mt-2"),
            html.Div(id='download-message', className="text-muted mt-1")
        ]),

        html.H3("3. RFM Table Preview", className="text-secondary mt-4"),
        dash_table.DataTable(id='data-preview', page_size=10, style_table={'overflowX': 'auto'}),

        html.Br(),
        html.Div(id='model-metrics', className="text-muted"),

        html.H3("4. Predict Customer Segment", className="text-secondary mt-5"),
        html.P("Enter RFM values for a new customer below to predict their segment.", className="fst-italic"),
        html.Ul([
            html.Li("Recency: Days since their last purchase (e.g., 7)"),
            html.Li("Frequency: Number of purchases made (e.g., 12)"),
            html.Li("Monetary: Total amount spent (e.g., 450.50)")
        ], className="text-muted"),
        html.Div([
            dcc.Input(id='recency-input', type='number', placeholder='Recency (days)', className="form-control"),
            dcc.Input(id='frequency-input', type='number', placeholder='Frequency (# purchases)', className="form-control mt-2"),
            dcc.Input(id='monetary-input', type='number', placeholder='Monetary ($)', className="form-control mt-2"),
            html.Button('Predict Segment', id='predict-button', n_clicks=0, className="btn btn-primary mt-2"),
            html.Div(id='prediction-output', className="mt-2 text-success fw-bold")
        ], className="mb-5")
    ], className="container")
])

@app.callback(
    Output('file-upload-output', 'children'),
    Output('data-preview', 'data'),
    Output('data-preview', 'columns'),
    Output('model-metrics', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return "No file uploaded", [], [], ""

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except:
        df = pd.read_csv(io.StringIO(decoded.decode('ISO-8859-1')))

    df_clean, X_scaled, scaler = clean_and_preprocess(df)
    model, labels, silhouette, best_model_name, model_scores_df = evaluate_models(X_scaled)
    df_clean['Cluster'] = labels

    store['df'] = df_clean
    store['model'] = model
    store['scaler'] = scaler
    store['best_model_name'] = best_model_name
    store['model_scores_df'] = model_scores_df

    columns = [{'name': i, 'id': i} for i in df_clean.columns]
    metrics = f"Best Model: {best_model_name} | Silhouette Score: {silhouette:.2f}"
    return f"Processed '{filename}'", df_clean.to_dict('records'), columns, metrics

@app.callback(
    Output('cluster-plot', 'figure'),
    Input('charts-tabs', 'value')
)
def update_chart(chart_type):
    return get_chart_figure(chart_type)

@app.callback(
    Output("download-image", "data"),
    Input("download-btn", "n_clicks"),
    State("charts-tabs", "value"),
    prevent_initial_call=True
)
def trigger_image_download(n, chart_type):
    fig = get_chart_figure(chart_type)
    return dcc.send_bytes(fig.to_image(format="png"), filename=f"chart_{chart_type}.png")

@app.callback(
    Output("download-table", "data"),
    Input("download-table-btn", "n_clicks"),
    prevent_initial_call=True
)
def trigger_csv_download(n_clicks):
    if store['df'] is not None:
        return dcc.send_data_frame(store['df'].to_csv, "rfm_export.csv")
    return dash.no_update

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('recency-input', 'value'),
    State('frequency-input', 'value'),
    State('monetary-input', 'value')
)
def predict_cluster(n_clicks, recency, frequency, monetary):
    if n_clicks == 0 or None in (recency, frequency, monetary):
        return ""
    if store['model'] is None or store['scaler'] is None or not hasattr(store['model'], 'predict'):
        return "Model not ready or unsupported for prediction."
    try:
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        input_scaled = store['scaler'].transform(input_df)
        cluster = store['model'].predict(input_scaled)[0]
        return f"Predicted Segment: Cluster {cluster}"
    except Exception as e:
        return f"Prediction error: {e}"

# Run Server: Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)


#if __name__ == '__main__':
#    app.run(debug=True)

# Run Server: Render
#if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 8050))  # Default to 8050 for local dev
#    app.run(host="0.0.0.0", port=port, debug=False)