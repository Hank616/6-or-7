import math
import random

class Perceptron:

    def __init__(self, size):

        self.w = [0] * size
        self.b = 0

    def score(self, x):

        return sum(
            wi * xi for wi, xi in zip(self.w, x)
        ) + self.b

    def predict(self, x):

        score = self.score(x)

        return 1 if score > 0 else -1

    def train(self, dataset, epochs):

        for epoch in range(epochs):

            random.shuffle(dataset)

            #print("\nEPOCH", epoch)

            for x, y in dataset:

                prediction = self.predict(x)

                #print("Prediction:", prediction, "Actual:", y)

                if prediction != y:

                    #print("UPDATING")

                    for i in range(len(self.w)):
                        self.w[i] += y * x[i]

                    self.b += y

                    #print("NEW BIAS:", self.b)
                    #print("FIRST 10 WEIGHTS:", self.w[:10])
    
    def probability(self, x):
        z = self.score(x)
        return 1 / (1 + math.exp(-z))