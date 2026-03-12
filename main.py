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
        model_path = os.path.join(os.getcwd(), "cardio_random_forest_model.pkl")

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

        # Feature Engineering (must match training)
        df["BMI"] = df["weight"] / (df["height"] / 100) ** 2
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

        # Ensure correct feature order
        df = df[columns]

        prediction = model.predict(df.values)[0]
        probability = model.predict_proba(df.values)[0][1]

        # Risk category
        if probability < 0.3:
            risk = "Low Risk"
        elif probability < 0.6:
            risk = "Moderate Risk"
        elif probability < 0.8:
            risk = "High Risk"
        else:
            risk = "Very High Risk"

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 3),
            "risk_level": risk,
            "result": "Cardiovascular Disease Detected"
            if prediction == 1 else "Healthy (No Cardio Disease)"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)