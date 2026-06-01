from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    features = [float(x) for x in request.form.values()]

    # Convert into numpy array
    final_features = np.array([features])

    # Make prediction
    prediction = model.predict(final_features)

    # Prediction probability
    probability = model.predict_proba(final_features)

    confidence = round(max(probability[0]) * 100, 2)

    # Result
    if prediction[0] == 1:
        result = f"Customer may churn ({confidence}% confidence)"
    else:
        result = f"Customer will stay ({confidence}% confidence)"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)