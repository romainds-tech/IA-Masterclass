use std::f64;
use rand::Rng;

#[derive(Clone, Copy)]
pub enum ActivationType {
    Threshold,
    Sigmoid,
    Tanh,
    Relu,
}

pub struct Perceptron {
    activation: ActivationType,
    biais: f64,
    pub weights: Vec<f64>,
    learning_rate: f64,
    output: f64,
    inputs: Vec<f64>,
    delta: f64,
}

impl Perceptron {
    pub fn new(nb_inputs: usize, activation: ActivationType) -> Perceptron {
        let mut rng = rand::thread_rng();
        Self {
            activation,
            biais: rng.gen_range(-1.0..1.0),
            weights: (0..nb_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            learning_rate: 0.1,
            output: 0.0,
            inputs: vec![0.0; nb_inputs],
            delta: 0.0,
        }
    }

    
    fn activate(&self, x: f64) -> f64 {
        match self.activation {
            ActivationType::Threshold => {
                if x >= 0.0 { 1.0 } else { 0.0 }
            },
            ActivationType::Sigmoid => {
                1.0 / (1.0 + f64::exp(-x))
            },
            ActivationType::Tanh => {
                x.tanh()
            },
            ActivationType::Relu => {
                f64::max(0.0, x)
            },
        }
    }

    fn dActivate(&self, x: f64) -> f64 {
        match self.activation {
            ActivationType::Threshold => {
                1.0
            },
            ActivationType::Sigmoid => {
                x * (1.0 - x)
            },
            ActivationType::Tanh => {
                1.0 - x.powi(2)
            },
            ActivationType::Relu => {
                if x > 0.0 { 1.0 } else { 0.0 }
            },
        }
    }

    pub fn predict(&mut self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());

        self.inputs.copy_from_slice(inputs);

        self.output = self.activate(
            inputs.iter()
                .zip(self.weights.iter())
                .map(|(input, weight)| input * weight)
                .sum::<f64>() + self.biais
        );
        self.output
    }

    pub fn train(&mut self, inputs: &[f64], target: f64) {
        assert_eq!(inputs.len(), self.weights.len());
    
        let prediction = self.predict(inputs);
        let error = target - prediction;
        
        self.delta = error * self.dActivate(self.output);
    
        let combination = self.learning_rate * error + self.delta;
    
        for (i, &input) in inputs.iter().enumerate() {
            self.weights[i] += input * combination;
        }
    
        self.biais += self.delta * self.learning_rate;
    }
    
}
