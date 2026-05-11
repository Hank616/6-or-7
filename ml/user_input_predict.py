import math
from pathlib import Path
import json
from perceptron import Perceptron

# TRAINING THE PERCEPTRON MODELS   
BASE_DIR = Path(__file__).resolve().parent
data_path_sixes = BASE_DIR.parent / "data" / "library.json"

with open(data_path_sixes) as f:
    raw_data = json.load(f)

dataset_sixes = []

for sample in raw_data:

    x = sample["pixels"]
    if sample["label"] == 6:
        y = 1
    else:  
        y = -1
    dataset_sixes.append((x, y))
p = Perceptron(63)
p.train(dataset_sixes, epochs=1000)
#print("\nFINAL WEIGHTS:")
#print(p.w)
#print("\nFINAL BIAS:")
#print(p.b)

# for sevens
dataset_sevens = []
for sample in raw_data:

    x = sample["pixels"]
    if sample["label"] == 7:
        y = 1
    else:  
        y = -1
    dataset_sevens.append((x, y))
k = Perceptron(63)
k.train(dataset_sevens, epochs=1000)
# print("\nFINAL WEIGHTS FOR SEVENS:")
# print(k.w)
# print("\nFINAL BIAS FOR SEVENS:")
# print(k.b)

# for nones
dataset_nones = []
for sample in raw_data:

    x = sample["pixels"]
    if sample["label"] == -1:
        y = 1
    else:  
        y = -1
    dataset_nones.append((x, y))
n = Perceptron(63)
n.train(dataset_nones, epochs=1000)
# print("\nFINAL WEIGHTS FOR NONES:")
# print(n.w)
# print("\nFINAL BIAS FOR NONES:")
# print(n.b)

data_path_testing = BASE_DIR.parent / "data" / "user_input.json"
test_data = []
with open(data_path_testing) as f:
    raw_data = json.load(f)
for sample in raw_data:

    x = sample["pixels"]

    test_data.append((x))

for x in test_data:
        ez6 = math.exp(p.score(x))
        ez7 = math.exp(k.score(x))
        ezn = math.exp(n.score(x))
        ez_total = ez6 + ez7 + ezn

        print("SIX: ",round(ez6/ez_total,5)*100, "% SEVEN: ", round(ez7/ez_total,5)*100, "% NONE: ", round(ezn/ez_total,5)*100, "%") 
        
        num1 = ez6/ez_total
        num2 = ez7/ez_total
        num3 = ezn/ez_total
        if num1 > num2 and num1 > num3:
            expectation = 6
        elif num2 > num1 and num2 > num3:
            expectation = 7
        else:            expectation = -1
        print("MODEL EXPECTATION:", expectation, "!")

