from typing import List
from perceptron import Perceptron, ActivationType


class Dense:
    def __init__(
        self,
        nbInput: int,
        nbHidden: int,
        hiddenAct: ActivationType,
        nbOutput: int,
        outputAct: ActivationType,
    ):
        self.hiddenLayer: List[Perceptron] = [
            Perceptron(nbInput, hiddenAct) for _ in range(nbHidden)
        ]
        self.outLayer: List[Perceptron] = [
            Perceptron(nbHidden, outputAct) for _ in range(nbOutput)
        ]
        self.outLayer: List[Perceptron] = [
            Perceptron(nbHidden, outputAct) for _ in range(nbOutput)
        ]

    def forward(self, inputs: List[float]) -> List[float]:
        hidden_output = [
            perceptron.forward(inputs) for perceptron in self.hiddenLayer
        ]
        return [
            perceptron.forward(hidden_output) for perceptron in self.outLayer
        ]

    def backward(self, deltas: List[float]):
        out_perceptron_delta = self.outLayer[0].backward(deltas[0])

        for perceptron in self.hiddenLayer:
            perceptron.backward(out_perceptron_delta)
