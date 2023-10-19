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
        self.error: float = 0.0

    def _activate(self, x: float) -> float:
        activations = {
            ActivationType.TRESHOLD: lambda x: 1.0 if x >= 0 else 0.0,
            ActivationType.SIGMOID: lambda x: 1.0 / (1.0 + math.exp(-x)),
            ActivationType.TANH: lambda x: math.tanh(x),
            ActivationType.RELU: lambda x: max(0.0, x),
        }
        return activations[self.act](x)

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

    def _calculate_error(self, expected: float) -> float:
        self.error = expected - self.output
        return self.error

    def _update_weights_and_biais(self):
        for i in range(len(self.weights)):
            self.weights[i] += self.inputs[i] * (
                self.learningRate * self.error
            )
        self.biais += self.learningRate * self.error

    def backward(self, expected: float):
        self._calculate_error(expected)
        self._update_weights_and_biais()

    def train(
        self, inputs: List[float], expected: float, learningRate: float = 0.01
    ) -> float:
        self.learningRate = learningRate
        self.forward(inputs)
        self.backward(expected)
        return abs(self.error)
