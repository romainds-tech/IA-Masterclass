import random
import math
from enum import Enum
from typing import List


class ActivationType(Enum):
    TRESHOLD = "Treshold"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    RELU = "ReLU"


class Perceptron:
    def __init__(self, nbInput: int, activation: ActivationType):
        self.act = activation
        self.biais = random.uniform(-1, 1)
        self.weights = [random.uniform(-1, 1) for _ in range(nbInput)]
        self.learningRate: float = 0.1
        self.inputs: List[float] = []
        self.output: float = 0.0
        self.delta: float = 0.0
        self.gradiants: [float] = [0 for _ in range(nbInput)]

    def _activate(self, x: float) -> float:
        activations = {
            ActivationType.TRESHOLD: lambda x: 1.0 if x >= 0 else 0.0,
            ActivationType.SIGMOID: lambda x: 1.0 / (1.0 + math.exp(-x)),
            ActivationType.TANH: lambda x: math.tanh(x),
            ActivationType.RELU: lambda x: max(0.0, x),
        }
        return activations[self.act](x)

    def _dActivate(
        self, x: float
    ) -> float:  # derivative of activation function
        derivatives = {
            ActivationType.TRESHOLD: 1.0,
            ActivationType.SIGMOID: lambda x: x * (1.0 - x),
            ActivationType.TANH: lambda x: 1.0 - math.tanh(x) ** 2,
            ActivationType.RELU: lambda x: 1.0 if x > 0 else 0.0,
        }
        return derivatives[self.act](x)

    def _dot(self, inputs: List[float]) -> float:
        return sum(
            input_val * weight
            for input_val, weight in zip(inputs, self.weights)
        )

    def forward(self, inputs: List[float]) -> float:
        if len(inputs) != len(self.weights):
            return float("NaN")
        self.inputs = inputs
        sop = self._dot(inputs)
        self.output = self._activate(sop + self.biais)
        return self.output

    def _update_weights_and_biais(self):
        for i in range(len(self.weights)):
            self.weights[i] += self.gradiants[i] * self.learningRate
        self.biais += self.delta * self.learningRate

    def backward(self, delta: float) -> float:
        self._calc_gradients(delta)
        self._update_weights_and_biais()
        return self.delta

    def _calc_gradients(self, delta: float):
        self.gradiants = []
        self.delta = self._dActivate(self.output) * delta
        for input in self.inputs:
            self.gradiants.append(input * self.delta)

    def train(
        self,
        inputs: List[float],
        expected: float,
        learningRate: float = 0.0001,
    ) -> float:
        self.learningRate = learningRate
        # predict
        self.forward(inputs)
        # calc error
        delta = expected - self.output
        # calc gradiants
        self.backward(delta)

        return abs(delta)
