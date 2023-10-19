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

    def forward(self, inputs: List[float]) -> List[float]:
        hiddenOutputs = [perceptron.forward(inputs) for perceptron in self.hiddenLayer]
        finalOutputs = [
            perceptron.forward(hiddenOutputs) for perceptron in self.outLayer
        ]

        return finalOutputs
