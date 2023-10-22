from perceptron import Perceptron, ActivationType


class Layer:
    def __init__(
        self,
        nbInputs: int,
        hiddenLayer: (int, ActivationType),
        outputLayer: (int, ActivationType),
    ):
        self.nbInputs = nbInputs
        self.hiddenLayer: [Perceptron] = [
            Perceptron(nbInputs, hiddenLayer[1]) for _ in range(hiddenLayer[0])
        ]
        self.outputLayer: [Perceptron] = [
            Perceptron(hiddenLayer[0], outputLayer[1])
            for _ in range(outputLayer[0])
        ]

    def predict(self, inputs: [float]) -> float:
        hiddenOutputs = [p.predict(inputs) for p in self.hiddenLayer]
        return [p.predict(hiddenOutputs) for p in self.outputLayer]

    def train(self, inputs: [float], targets: [float]):
        # 1. Forward pass
        hiddenOutputs = [p.predict(inputs) for p in self.hiddenLayer]
        finalOutputs = [p.predict(hiddenOutputs) for p in self.outputLayer]

        # 2. Calculate the output layer error
        outputErrors = [
            target - output for target, output in zip(targets, finalOutputs)
        ]

        # 3. Update the output layer weights
        for i, perceptron in enumerate(self.outputLayer):
            perceptron.train(hiddenOutputs, targets[i])

        # 4. Calculate the hidden layer error
        hiddenErrors = [0] * len(self.hiddenLayer)
        for i, perceptron in enumerate(self.hiddenLayer):
            error = 0
            for j, outPerceptron in enumerate(self.outputLayer):
                error += outPerceptron.weights[i] * outputErrors[j]
            hiddenErrors[i] = error

        # 5. Update the hidden layer weights
        for i, perceptron in enumerate(self.hiddenLayer):
            perceptron.train(
                inputs, hiddenErrors[i] + hiddenOutputs[i]
            )  # Adding current output to match perceptron's training method
