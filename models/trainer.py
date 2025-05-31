import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils.feature_engineering import prepare_features


def train_model():
    df = pd.read_csv("data/raw/game_results.csv")
    df = prepare_features(df)
    X = df[["home_advantage", "momentum"]]
    y = df["result"].map({"W": 1, "L": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    joblib.dump(model, "models/baseline_model.pkl")


if __name__ == "__main__":
    train_model()
