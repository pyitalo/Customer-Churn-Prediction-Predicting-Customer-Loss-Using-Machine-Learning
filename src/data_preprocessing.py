import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Remove customerID se existir
    df = df.drop(columns=["customerID"], errors="ignore")

    # Converter todas as colunas numéricas para numerico com coerce
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Converte TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Preencher qualquer NaN numérico com a mediana
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fazer label encoding em colunas categóricas
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # DEBUG – ver se ainda existe NaN
    print("\n🔍 Checando NaNs após preprocessamento:")
    print(df.isna().sum())

    # Divide X e y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

