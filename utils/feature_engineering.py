def prepare_features(df):
    df["home_advantage"] = df["venue"].apply(lambda x: 1 if x == "Home" else 0)
    df["momentum"] = df["result"].map({"W": 1, "L": 0}).rolling(5).mean().fillna(0)
    return df
