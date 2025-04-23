import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
import base64
import io

app = Dash(__name__)
app.title = 'Customer Segmentation (RFM)' 

df_global = None
model_global = None
scaler_global = None

# Layout
app.layout = html.Div([
    html.H1("Customer Segmentation with RFM", style={'textAlign': 'center'}),

    html.H2("1. Upload Transactions CSV (.csv with InvoiceNo, Quantity, UnitPrice, InvoiceDate, CustomerID)"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='file-upload-output'),

    html.H2("2. Predict Customer Segment (Input RFM Values)"),
    html.Div([
        html.Label('Recency:'),
        dcc.Input(id='recency-input', type='number', min=0),
        html.Label('Frequency:'),
        dcc.Input(id='frequency-input', type='number', min=0),
        html.Label('Monetary:'),
        dcc.Input(id='monetary-input', type='number', min=0, step=0.01),
        html.Button('Predict Segment', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output', style={'marginTop': '10px', 'fontWeight': 'bold'})
    ], style={'display': 'flex', 'gap': '10px'}),

    html.H2("3. Cluster Visualization"),
    dcc.Graph(id='cluster-plot'),

    html.H2("4. RFM Dataset Preview"),
    dash_table.DataTable(id='data-preview', page_size=10, style_table={'overflowX': 'auto'}),

    html.Br(),
    html.Div(id='model-metrics'),
])

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
    rfm = rfm[~((rfm[['Recency', 'Frequency', 'Monetary']] < (Q1 - 1.5 * IQR)) |
                (rfm[['Recency', 'Frequency', 'Monetary']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm, X_scaled, scaler

def train_kmeans(X_scaled, k=4):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return model, labels, score

@app.callback(
    Output('file-upload-output', 'children'),
    Output('cluster-plot', 'figure'),
    Output('data-preview', 'data'),
    Output('data-preview', 'columns'),
    Output('model-metrics', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global df_global, model_global, scaler_global

    if contents is None:
        return "No file uploaded", {}, [], [], ""

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except:
        df = pd.read_csv(io.StringIO(decoded.decode('ISO-8859-1')))

    df_clean, X_scaled, scaler = clean_and_preprocess(df)
    model, clusters, silhouette = train_kmeans(X_scaled)
    df_clean['Cluster'] = clusters
    df_clean.to_csv('predicted_clusters.csv', index=False)

    df_global = df_clean
    model_global = model
    scaler_global = scaler

    fig = px.scatter(
        df_clean, x='Recency', y='Monetary', color='Cluster', hover_data=['Frequency'],
        title='Customer Clusters (RFM)')

    columns = [{'name': i, 'id': i} for i in df_clean.columns]
    metrics = f"Silhouette Score: {silhouette:.2f} | Saved to 'predicted_clusters.csv'"

    return f"Processed '{filename}'", fig, df_clean.to_dict('records'), columns, metrics

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
    if model_global is None or scaler_global is None:
        return "Please upload a dataset first."

    input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    input_scaled = scaler_global.transform(input_df)
    cluster = model_global.predict(input_scaled)[0]
    return f"Predicted Segment: {cluster}"

if __name__ == '__main__':
    app.run(debug=True)
