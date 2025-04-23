# 2. preporcess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

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


