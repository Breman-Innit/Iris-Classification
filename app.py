
from pathlib import Path

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, static_folder="public", static_url_path="")

# Load the trained model from the project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.pkl"

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request data."}), 400

        measurements = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]

        prediction = model.predict([measurements])[0]
        probabilities = model.predict_proba([measurements])[0]

        result = {
            "prediction": str(prediction),
            "confidence": round(float(max(probabilities)) * 100, 2)
        }

        return jsonify(result)

    except (ValueError, TypeError, KeyError):
        return jsonify({
            "error": "Please provide four valid numerical measurements."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "An error occurred while making the prediction."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
