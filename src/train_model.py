import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import load_dataset, preprocess_data

def train_and_save_model():
    df = load_dataset("data/raw/Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    joblib.dump(model, "models/churn_model.pkl")
    print("\nModel saved at models/churn_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
