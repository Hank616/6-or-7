from pathlib import Path
import json
from perceptron import Perceptron

BASE_DIR = Path(__file__).resolve().parent
data_path_sixes = BASE_DIR.parent / "data" / "library.json"

with open(data_path_sixes) as f:
    raw_data = json.load(f)

dataset_sixes = []

for sample in raw_data:

    x =  Perceptron.normalize(sample["pixels"])
    if sample["label"] == 6:
        y = 1
    else:  
        y = -1
    dataset_sixes.append((x, y))

p = Perceptron(63)

p.train(dataset_sixes, epochs=1000)

print("\nFINAL WEIGHTS FOR SIXES:")
print(p.w)

print("\nFINAL BIAS FOR SIXES:")
print(p.b)

weights = {
    "w": p.w,
    "b": p.b
}

# for sevens
dataset_sevens = []

for sample in raw_data:

    x = Perceptron.normalize(sample["pixels"])
    if sample["label"] == 7:
        y = 1
    else:  
        y = -1
    dataset_sevens.append((x, y))

k = Perceptron(63)

k.train(dataset_sevens, epochs=1000)

print("\nFINAL WEIGHTS FOR SEVENS:")
print(k.w)

print("\nFINAL BIAS FOR SEVENS:")
print(k.b)

# for nones
dataset_nones = []

for sample in raw_data:

    x = Perceptron.normalize(sample["pixels"])
    if sample["label"] == -1:
        y = 1
    else:  
        y = -1
    dataset_nones.append((x, y))

n = Perceptron(63)

n.train(dataset_nones, epochs=1000)

print("\nFINAL WEIGHTS FOR NONES:")
print(n.w)

print("\nFINAL BIAS FOR NONES:")
print(n.b)
