# 6 or 7 AI

A simple handwritten digit classifier built from scratch using perceptrons and Flask.

The user draws on a 7×9 pixel canvas, and the model predicts:
- 6
- 7
- None

No ML libraries are used.

---

# Project Structure

```txt
6-or-7/

├── frontend/
│     ├── run_it.html
│     ├── script.js
│     └── style.css
│
├── backend/
│     ├── app.py
│     ├── perceptron.py
│     ├── testing.py
│     └── train.py
│
├── data/
│     ├── library.json
│     └── testing.json
│
└── README.md
```

---

# Install

```bash
pip install flask flask-cors
```

---

# Run the Backend

Open terminal:

```bash
cd ml
python app.py
```

You should see:

```txt
Running on http://127.0.0.1:5000
```

Keep this terminal open.

---

# Run the Frontend

Open:

```txt
frontend/run_it.html
```

in your browser.

Draw a digit and click:

```txt
Predict
```

---

# Test Overall Accuracy  
  
$$\color{red}{\textbf{Current Accuracy: 86\\%}\sim\textbf{93\\%}}$$

Run:

```bash
cd ml
python testing.py
```

Example:

```txt
SIX:  0.0 % SEVEN:  99.331 % NONE:  0.6689999999999999 % ACTUAL: 7
SIX:  0.0 % SEVEN:  0.247 % NONE:  99.753 % ACTUAL: -1
SIX:  0.001 % SEVEN:  0.001 % NONE:  99.99900000000001 % ACTUAL: 6
WRONG PREDICTION!
SIX:  0.0 % SEVEN:  0.0 % NONE:  100.0 % ACTUAL: 7
ACCURACY: 93.21 %
```

---

# Basic Math

Each perceptron computes a score:

```math
z = w \cdot x + b
```

Where:
- `w` = weight vector
- `x` = pixel vector
- `b` = bias

The scores are converted into probabilities using softmax:

```math
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

The model predicts the class with the highest probability.

---

# Notes

This project was built for learning:
- perceptrons
- hyperplanes
- softmax
- linear classification
- frontend/backend communication
- basic machine learning systems
