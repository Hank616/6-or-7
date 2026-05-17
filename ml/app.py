from flask import Flask, request, jsonify
from flask_cors import CORS

import math
from pathlib import Path
import json

from perceptron import Perceptron


# =====================================
# FLASK SETUP
# =====================================

app = Flask(__name__)

CORS(app)


# =====================================
# LOAD TRAINING DATA
# =====================================

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR.parent / "data" / "library.json"

with open(data_path) as f:

    raw_data = json.load(f)


# =====================================
# TRAIN SIX MODEL
# =====================================

dataset_sixes = []

for sample in raw_data:

    x = Perceptron.normalize(sample["pixels"])

    if sample["label"] == 6:
        y = 1
    else:
        y = -1

    dataset_sixes.append((x, y))


p = Perceptron(63)

p.train(dataset_sixes, epochs=1000)


# =====================================
# TRAIN SEVEN MODEL
# =====================================

dataset_sevens = []

for sample in raw_data:

    x =  Perceptron.normalize(sample["pixels"])

    if sample["label"] == 7:
        y = 1
    else:
        y = -1

    dataset_sevens.append((x, y))


k = Perceptron(63)

k.train(dataset_sevens, epochs=1000)


# =====================================
# TRAIN NONE MODEL
# =====================================

dataset_nones = []

for sample in raw_data:

    x =  Perceptron.normalize(sample["pixels"])

    if sample["label"] == -1:
        y = 1
    else:
        y = -1

    dataset_nones.append((x, y))


n = Perceptron(63)

n.train(dataset_nones, epochs=1000)


print("\nMODELS TRAINED SUCCESSFULLY")


# =====================================
# SOFTMAX FUNCTION
# =====================================

def softmax(scores):

    exps = [math.exp(s) for s in scores]

    total = sum(exps)

    return [e / total for e in exps]


# =====================================
# PREDICT ROUTE
# =====================================

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json

    x =  Perceptron.normalize(data["pixels"])


    # raw scores
    z6 = p.score(x)

    z7 = k.score(x)

    zn = n.score(x)


    # softmax probabilities
    probs = softmax([z6, z7, zn])


    six_prob = probs[0]

    seven_prob = probs[1]

    none_prob = probs[2]


    # prediction
    if six_prob > seven_prob and six_prob > none_prob:

        expectation = 6

    elif seven_prob > six_prob and seven_prob > none_prob:

        expectation = 7

    else:

        expectation = -1


    print("\n======================")
    print("USER INPUT:")
    print(x)

    print("\nSCORES:")
    print("6:", z6)
    print("7:", z7)
    print("NONE:", zn)

    print("\nPROBABILITIES:")
    print("6:", round(six_prob * 100, 2), "%")
    print("7:", round(seven_prob * 100, 2), "%")
    print("NONE:", round(none_prob * 100, 2), "%")

    print("\nPREDICTION:", expectation)
    print("======================\n")


    # send JSON back to frontend
    return jsonify({

        "six": six_prob,

        "seven": seven_prob,

        "none": none_prob,

        "prediction": expectation,


        # scores
        "six_score": z6,

        "seven_score": z7,

        "none_score": zn,


        # bias
        "six_bias": p.b,

        "seven_bias": k.b,

        "none_bias": n.b,


        # weights
        "six_weights": p.w,

        "seven_weights": k.w,

        "none_weights": n.w
    })


# =====================================
# RETRAIN ROUTE
# =====================================

@app.route("/retrain", methods=["POST"])

def retrain():

    global p, k, n

    p = Perceptron(63)
    p.train(dataset_sixes, epochs=1000)

    k = Perceptron(63)
    k.train(dataset_sevens, epochs=1000)

    n = Perceptron(63)
    n.train(dataset_nones, epochs=1000)

    print("\nMODELS RETRAINED")

    return jsonify({

        "message": "Models retrained successfully!"
    })


# =====================================
# START SERVER
# =====================================

if __name__ == "__main__":

    app.run(debug=True)