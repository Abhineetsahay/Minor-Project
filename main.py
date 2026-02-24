from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
# Load trained model
with open("cardio_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = data["columns"]


@app.route("/")
def home():
    return "Cardio Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()

    # Convert input to dataframe
    df = pd.DataFrame([input_data])
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2

    # Ensure correct column order
    df = df[columns]

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(probability), 3),
        "result": "Cardiovascular Disease Detected" if prediction == 1 else "Healthy (No Cardio Disease)"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)