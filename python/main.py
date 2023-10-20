from perceptron import ActivationType
from dense import Dense

"""
 This is a simple dataset to test the perceptron
 First value is the temperature, the seconde is the humidity and the last one
 is the wind speed
 The expected value is 1.0 if it's good condition to wearing a jacket, 0.0
 otherwise

 dataset = [
    # Temperature, rain risk, wind | response
    ([2, 60, 20], 1.0),
    ([5, 30, 40], 1.0),
    ([8, 40, 10], 1.0),
    ([12, 80, 10], 1.0),
    ([9, 0, 50], 1.0),
    ([23, 10, 10], 0.0),  # Used for testing
    ([20, 0, 40], 0.0),
    ([19, 5, 20], 0.0),
    ([22, 10, 40], 0.0),
    ([29, 0, 0], 0.0),
]
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

epoch = 10000

dense = Dense(
    nbInput=3,
    nbHidden=6,
    hiddenAct=ActivationType.RELU,
    nbOutput=1,
    outputAct=ActivationType.SIGMOID,
)

for e in range(epoch):
    meanError = 0.0
    for d in dataset:
        # Prediction
        readyInput = [d[0][0] / 100.0, d[0][1] / 100.0, d[0][2] / 100.0]

        pred = dense.forward(inputs=readyInput)
        # Uncomment to print weights before and after training
        # print("Before training")
        # print([p.weights for p in dense.hiddenLayer])
        # print([p.weights for p in dense.outLayer])
        # print(pred)
        # Calc Error (delta)
        error = d[1] - pred[0]
        dense.backward(deltas=[error])
        # Uncomment to print weights after training
        # print("After training")
        # print([p.weights for p in dense.hiddenLayer])
        # print([p.weights for p in dense.outLayer])
        meanError += abs(error)
    meanError /= len(dataset)
    print(meanError)


"""

# print('perceptron ...')
p = Perceptron(3, ActivationType.TRESHOLD)
epoch = 1000

print("before training")

pred = p.forward([2.0, 3.0, 4.0])
print(pred)

print("training ...")

for e in range(epoch):
    currentEpochError = 0.0

    for i in dataset:
        currentEpochError += p.train(i[0], i[1])

    meanError = currentEpochError / len(dataset)

    print(meanError)

print("after training")
print(p.weights)


"""
