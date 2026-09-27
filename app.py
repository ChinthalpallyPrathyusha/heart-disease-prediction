from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd


# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------
model = joblib.load("heart_disease_model (2).pkl")
scaler = joblib.load("scaler (1).pkl")
print("Model and scaler loaded successfully.")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get data sent from frontend
        data = request.get_json()

        # --------------------------------------------------
        # Create dataframe using the same original features
        # --------------------------------------------------

        input_data = pd.DataFrame([{
            "Age": float(data["age"]),
            "Sex": data["sex"],
            "RestingBP": float(data["resting_bp"]),
            "Cholesterol": float(data["cholesterol"]),
            "FastingBS": int(data["fasting_bs"]),
            "MaxHR": float(data["max_hr"]),
            "ExerciseAngina": data["exercise_angina"],
            "Oldpeak": float(data["oldpeak"]),
            "ChestPainType": data["chest_pain"],
            "RestingECG": data["resting_ecg"],
            "ST_Slope": data["st_slope"]
        }])


        # --------------------------------------------------
        # Encode categorical variables
        # --------------------------------------------------

        input_data["Sex"] = input_data["Sex"].map({
            "M": 1,
            "F": 0
        })

        input_data["ExerciseAngina"] = input_data["ExerciseAngina"].map({
            "Y": 1,
            "N": 0
        })


        # One-hot encoding
        input_data = pd.get_dummies(
            input_data,
            columns=["ChestPainType", "RestingECG", "ST_Slope"],
            drop_first=True
        )


        # --------------------------------------------------
        # Make sure all model columns are present
        # --------------------------------------------------

        required_columns = [
            "Age",
            "Sex",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "MaxHR",
            "ExerciseAngina",
            "Oldpeak",
            "ChestPainType_ATA",
            "ChestPainType_NAP",
            "ChestPainType_TA",
            "RestingECG_Normal",
            "RestingECG_ST",
            "ST_Slope_Flat",
            "ST_Slope_Up"
        ]

        for column in required_columns:
            if column not in input_data.columns:
                input_data[column] = 0


        # Keep exactly the same feature order
        input_data = input_data[required_columns]


        # --------------------------------------------------
        # Scale continuous features
        # --------------------------------------------------

        continuous_columns = [
            "Age",
            "RestingBP",
            "Cholesterol",
            "MaxHR",
            "Oldpeak"
        ]

        input_data[continuous_columns] = scaler.transform(
            input_data[continuous_columns]
        )


        # --------------------------------------------------
        # Make prediction
        # --------------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1] * 100


        # --------------------------------------------------
        # Result text
        # --------------------------------------------------

        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"


        # --------------------------------------------------
        # Send result to frontend
        # --------------------------------------------------

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "result": result
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )