use crate::perceptron::{Perceptron, ActivationType};

pub struct Dense {
    hidden_layer: Vec<Perceptron>,
    out_layer: Vec<Perceptron>,
}

impl Dense {
    pub fn new(
        nb_input: usize,
        nb_hidden: usize,
        hidden_act: ActivationType,
        nb_output: usize,
        output_act: ActivationType,
    ) -> Self {
        let hidden_layer = (0..nb_hidden)
            .map(|_| Perceptron::new(nb_input, hidden_act.clone()))
            .collect();

        let out_layer = (0..nb_output)
            .map(|_| Perceptron::new(nb_hidden, output_act.clone()))
            .collect();

        Dense {
            hidden_layer,
            out_layer,
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let hidden_outputs: Vec<f64> = self
            .hidden_layer
            .iter_mut()
            .map(|perceptron| perceptron.forward(&inputs))
            .collect();

        let final_outputs: Vec<f64> = self
            .out_layer
            .iter_mut()
            .map(|perceptron| perceptron.forward(&hidden_outputs))
            .collect();

        final_outputs
    }
}
