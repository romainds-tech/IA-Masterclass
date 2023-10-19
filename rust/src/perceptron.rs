use rand::Rng;

#[derive(Clone, Copy)]
pub enum ActivationType {
    TRESHOLD,
    SIGMOID,
    TANH,
    RELU,
}

pub struct Perceptron {
    act: ActivationType,
    biais: f64,
    weights: Vec<f64>,
    learning_rate: f64,
    inputs: Vec<f64>,
    output: f64,
    error: f64,
}

impl Perceptron {
    pub fn new(nb_input: usize, activation: ActivationType) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..nb_input).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self {
            act: activation,
            biais: rng.gen_range(-1.0..1.0),
            weights,
            learning_rate: 0.1,
            inputs: Vec::new(),
            output: 0.0,
            error: 0.0,
        }
    }

    fn activate(&self, x: f64) -> f64 {
        match self.act {
            ActivationType::TRESHOLD => if x >= 0.0 { 1.0 } else { 0.0 },
            ActivationType::SIGMOID => 1.0 / (1.0 + (-x).exp()),
            ActivationType::TANH => x.tanh(),
            ActivationType::RELU => x.max(0.0),
        }
    }

    fn dot(&self, inputs: &[f64]) -> f64 {
        self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum()
    }

    pub fn forward(&mut self, inputs: &[f64]) -> f64 {
        if inputs.len() != self.weights.len() {
            return f64::NAN;
        }
        self.inputs = inputs.to_vec();
        let sop = self.dot(&inputs);
        self.output = self.activate(sop + self.biais);
        self.output
    }

    fn calculate_error(&mut self, expected: f64) -> f64 {
        self.error = expected - self.output;
        self.error
    }

    fn update_weights_and_biais(&mut self) {
        for (weight, input) in self.weights.iter_mut().zip(&self.inputs) {
            *weight += input * self.learning_rate * self.error;
        }
        self.biais += self.learning_rate * self.error;
    }

    fn backward(&mut self, expected: f64) {
        self.calculate_error(expected);
        self.update_weights_and_biais();
    }

    pub fn train(&mut self, inputs: &[f64], expected: f64, learning_rate: f64) -> f64 {
        self.learning_rate = learning_rate;
        self.forward(inputs);
        self.backward(expected);
        self.error.abs()
    }
}