from flask import Flask, render_template, request, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename
import joblib
from utils.feature_engineering import prepare_features

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "raw")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            df = pd.read_csv(file_path)
            df = prepare_features(df)
            model = joblib.load("models/baseline_model.pkl")
            prediction = model.predict(df[["home_advantage", "momentum"]])
            prediction = prediction.tolist()
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
