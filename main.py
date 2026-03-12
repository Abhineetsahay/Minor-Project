from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS
import os

app = Flask(__name__, template_folder="templates")
CORS(app)

model = None
columns = None


def load_model():
    global model, columns

    if model is None:
        model_path = os.path.join(os.getcwd(), "cardio_best_model.pkl")

        if not os.path.exists(model_path):
            raise Exception("Model file not found!")

        data = joblib.load(model_path)

        model = data["model"]
        columns = data["columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()

        input_data = request.get_json()

        df = pd.DataFrame([input_data])

        # BMI feature engineering
        df["BMI"] = df["weight"] / (df["height"] / 100) ** 2

        # Ensure correct column order
        df = df[columns]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 3),
            "result": "Cardiovascular Disease Detected"
            if prediction == 1 else "Healthy (No Cardio Disease)"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)