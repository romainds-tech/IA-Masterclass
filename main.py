from perceptron import ActivationType
from dense import Dense

"""
 This is a simple dataset to test the perceptron
 First value is the temperature, the seconde is the humidity and the last one
 is the wind speed
 The expected value is 1.0 if it's good condition to wearing a jacket, 0.0
 otherwise
"""

dataset: [([float], float)] = [
    ([2, 60, 20], 1.0),
    ([2, 70, 10], 1.0),
    ([2, 80, 15], 1.0),
    ([12, 50, 35], 1.0),
    ([12, 30, 80], 1.0),
    #
    ([40, 10, 20], 0.0),
    ([35, 10, 0], 0.0),
    ([28, 5, 20], 0.0),
    ([32, 10, 5], 0.0),
    ([30, 15, 2], 0.0),
]


""" This code is basic usage of the perceptron
epoch = 10
p = Perceptron(3, ActivationType.Treshold)
print(p.weights)

for e in range(epoch):
    currentEpoch = 0.0
    for i in dataset:
        currentEpoch += p.train(i[0], i[1], 0.001)
    print(currentEpoch / len(dataset))

print(p.weights)
"""


# This code is basic usage of the dense layer whitout stochastic gradient
# descent
dense = Dense(3, 2, ActivationType.RELU, 2, ActivationType.SIGMOID)
pred = dense.forward([1.0, 2.0, 3.0])

print(pred)
