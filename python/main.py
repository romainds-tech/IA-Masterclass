import time
from perceptron import ActivationType
from layer import Layer

# from layer import Layer

if __name__ == "__main__":
    start = time.time()

    dataset = [
        ([0.12, 0.90, 0.10], 1),
        ([0.10, 0.70, 0.40], 1),
        ([0.14, 0.40, 0.20], 1),
        ([0.01, 0.60, 0.60], 1),
        ([0.08, 0.50, 0.30], 1),
        #
        ([0.25, 0.10, 0.10], 0),
        ([0.30, 0.15, 0.10], 0),
        ([0.28, 0.05, 0.30], 0),
        ([0.21, 0.0, 0.0], 0),
        ([0.40, 0.10, 0.10], 0),
    ]

    epoch = 100

    allErrors = []

    """

    for i in range(1000):
        p = Perceptron(3, ActivationType.THRESHOLD)

        errorMean = 0
        for inputs, target in dataset:
            errorMean += abs(target - p.predict(inputs))
        errorMean /= len(dataset)

        #  print(f"error mean before training: {errorMean}")
        # print("training...")

        for e in range(epoch):
            for inputs, target in dataset:
                p.train(inputs, target)

        errorMean = 0
        for inputs, target in dataset:
            errorMean += abs(target - p.predict(inputs))
        errorMean /= len(dataset)

        # print(f"error mean after training: {errorMean}")

        allErrors.append(errorMean)

    print("Moyenne des erreurs: ", sum(allErrors) / len(allErrors))

    """

    # mean error before training

    for e in range(500):
        layer = Layer(
            3, (9, ActivationType.RELU), (1, ActivationType.THRESHOLD)
        )

        for e in range(epoch):
            for inputs, target in dataset:
                layer.train(inputs, [target])

        errorMean = 0
        for inputs, target in dataset:
            errorMean += abs(target - layer.predict(inputs)[0])
        errorMean /= len(dataset)

        allErrors.append(errorMean)

    print("Moyenne des erreurs: ", sum(allErrors) / len(allErrors))

    print(f"Temps écoulé: {time.time() - start} secondes")
