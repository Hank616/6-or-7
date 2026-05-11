from flask import Flask, request, jsonify
from flask_cors import CORS

import json
import math
from pathlib import Path

app = Flask(__name__)

CORS(app)

BASE_DIR = Path(__file__).resolve().parent


# =========================
# LOAD MODEL
# =========================

def load_model(filename):

    path = BASE_DIR.parent / "models" / filename

    with open(path) as f:

        model = json.load(f)

    return model["w"], model["b"]


# load all models
six_w, six_b = load_model("six_weights.json")

seven_w, seven_b = load_model("seven_weights.json")

none_w, none_b = load_model("none_weights.json")


# =========================
# SCORE FUNCTION
# =========================

def score(x, w, b):

    return sum(
        xi * wi for xi, wi in zip(x, w)
    ) + b


# =========================
# SOFTMAX
# =========================

def softmax(scores):

    exps = [math.exp(s) for s in scores]

    total = sum(exps)

    return [e / total for e in exps]


# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json

    pixels = data["pixels"]


    # model scores
    z6 = score(pixels, six_w, six_b)

    z7 = score(pixels, seven_w, seven_b)

    zNone = score(pixels, none_w, none_b)


    # probabilities
    probs = softmax([z6, z7, zNone])


    return jsonify({

        "six": probs[0],
        "seven": probs[1],
        "none": probs[2],

        "six_score": z6,
        "seven_score": z7,
        "none_score": zNone,

        "six_bias": six_b,
        "seven_bias": seven_b,
        "none_bias": none_b,

        # show only first 15 weights
        "six_weights": six_w[:15],
        "seven_weights": seven_w[:15],
        "none_weights": none_w[:15]
    })


# =========================
# RETRAIN ROUTE
# =========================

@app.route("/retrain", methods=["POST"])

def retrain():

    # run training file
    import train

    return jsonify({
        "message":"Model retrained successfully!"
    })


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    app.run(debug=True)