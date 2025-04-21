import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from dash import Dash, dcc, html, Input, Output

# ========== Load and Preprocess Data ==========
df = pd.read_csv("../data/cleaned_data.csv")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'UnitPrice': lambda x: (x * df.loc[x.index, 'Quantity']).sum()
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# ========== Model Building ==========
# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['KMeans'] = kmeans.fit_predict(rfm_scaled)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=4)
rfm['Agglomerative'] = agg.fit_predict(rfm_scaled)

# DBSCAN
db = DBSCAN(eps=1.5, min_samples=5)
rfm['DBSCAN'] = db.fit_predict(rfm_scaled)

# Silhouette Scores
scores = {
    'KMeans': silhouette_score(rfm_scaled, rfm['KMeans']),
    'Agglomerative': silhouette_score(rfm_scaled, rfm['Agglomerative']),
    'DBSCAN': silhouette_score(rfm_scaled, rfm['DBSCAN']) if len(set(rfm['DBSCAN'])) > 1 else -1
}
scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Silhouette Score']).reset_index()
scores_df.rename(columns={'index': 'Model'}, inplace=True)

# ========== Dash App ==========
app = Dash(__name__)
app.title = "Customer Segmentation Dashboard"

app.layout = html.Div([
    html.H1("Customer Segmentation", style={'textAlign': 'center'}),

    html.H3("Silhouette Score Comparison"),
    dcc.Graph(
        id="silhouette-score-bar",
        figure=px.bar(scores_df, x="Model", y="Silhouette Score", title="Model Comparison")
    ),

    html.H3("Cluster Visualization"),
    html.Label("Choose Clustering Model:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model, 'value': model} for model in ['KMeans', 'Agglomerative', 'DBSCAN']],
        value='KMeans'
    ),

    dcc.Graph(id='cluster-graph'),

    html.H3("Cluster Summary"),
    html.Div(id='cluster-summary')
])

@app.callback(
    Output('cluster-graph', 'figure'),
    Output('cluster-summary', 'children'),
    Input('model-dropdown', 'value')
)
def update_cluster_view(model_choice):
    fig = px.scatter_3d(
        rfm, x='Recency', y='Frequency', z='Monetary',
        color=rfm[model_choice].astype(str),
        title=f"{model_choice} Clustering",
        labels={'color': 'Cluster'}
    )

    summary = rfm.groupby(model_choice).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Num_Customers'}).reset_index()

    return fig, html.Div([
        html.H5(f"{model_choice} Cluster Summary"),
        dcc.Graph(
            figure=px.bar(summary, x=model_choice, y='Num_Customers',
                          title='Number of Customers per Cluster')
        )
    ])

if __name__ == "__main__":
    app.run(debug=True)
