# 2. preporcess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Clean and compute RFM
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

    # Scale RFM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    return X_scaled, rfm, scaler

