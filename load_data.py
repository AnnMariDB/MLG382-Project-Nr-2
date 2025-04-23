# 1. pythong load_data.py

import pandas as pd

def load_data(filepath="customerSegData.csv"):
    try:
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        print(f"[INFO] Loaded data with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("[ERROR] File not found. Please check the path.")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading data: {e}")
        return None
