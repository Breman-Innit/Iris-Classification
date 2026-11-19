
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Iris classification model
model = joblib.load("iris_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get the four flower measurements
        measurements = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]

        # Make a prediction
        prediction = model.predict([measurements])[0]

        # Get prediction probabilities
        probabilities = model.predict_proba([measurements])[0]

        result = {
            "prediction": str(prediction),
            "confidence": round(float(max(probabilities)) * 100, 2)
        }

        return jsonify(result)

    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Please provide four valid numerical measurements."}), 400

    except Exception:
        return jsonify({"error": "An error occurred while making the prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True)
