from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__, template_folder="templates")
CORS(app)

model = None
columns = None

def load_model():
    global model, columns
    if model is None:
        model_path = os.path.join(os.getcwd(), "cardio_model.pkl")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        columns = data["columns"]

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    
    input_data = request.get_json()

    df = pd.DataFrame([input_data])
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2

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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)