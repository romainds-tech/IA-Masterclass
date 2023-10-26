from enum import Enum
from typing import List
import random
import math


class ActivationType(Enum):
    THRESHOLD = "Threshold"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    RELU = "ReLU"


class Perceptron:
    def __init__(self, nbInputs: int, activation: ActivationType):
        self.activation = activation
        self.weights = [random.uniform(-1, 1) for _ in range(nbInputs)]
        self.bias = random.uniform(-1, 1)
        self.learningRate = 0.1
        self.output = 0.0
        self.delta: float = 0.0
        self.inputs: List[float] = []
        self.total_input: float = 0.0

    def _activate(self, x: float) -> float:
        if self.activation == ActivationType.THRESHOLD:
            return 1.0 if x >= 0 else 0.0
        if self.activation == ActivationType.SIGMOID:
            return 1.0 / (1.0 + math.exp(-x))
        if self.activation == ActivationType.TANH:
            return math.tanh(x)
        if self.activation == ActivationType.RELU:
            return max(0.0, x)

    def _d_activate(self, x: float) -> float:
        if self.activation == ActivationType.THRESHOLD:
            return 0.0
        if self.activation == ActivationType.SIGMOID:
            y = self._activate(x)
            return y * (1.0 - y)
        if self.activation == ActivationType.TANH:
            return 1.0 - math.tanh(x) ** 2
        if self.activation == ActivationType.RELU:
            return 1.0 if x > 0 else 0.0

    def predict(self, inputs: List[float]) -> float:
        self.inputs = inputs
        self.total_input = (
            sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        )
        return self._activate(self.total_input)

    def train(self, inputs: List[float], target: float):
        prediction = self.predict(inputs)
        error = target - prediction

        # calculate gradients
        self.delta = error * self._d_activate(self.total_input)

        # adjust weights and bias
        for i in range(len(self.weights)):
            self.weights[i] += (
                self.delta * inputs[i]
            ) + self.learningRate * error * inputs[i]
        self.bias += self.delta * self.learningRate
