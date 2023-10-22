use crate::perceptron::{Perceptron, ActivationType};


pub struct Layer {
    nb_inputs: usize,
    hidden_layer: Vec<Perceptron>,
    output_layer: Vec<Perceptron>,
}

impl Layer {
    pub fn new(
        nb_inputs: usize,
        hidden_layer: (usize, ActivationType),
        output_layer: (usize, ActivationType),
    ) -> Self {
        let hidden = (0..hidden_layer.0)
            .map(|_| Perceptron::new(nb_inputs, hidden_layer.1))
            .collect();
        let output = (0..output_layer.0)
            .map(|_| Perceptron::new(hidden_layer.0, output_layer.1))
            .collect();

        Self {
            nb_inputs,
            hidden_layer: hidden,
            output_layer: output,
        }
    }

    pub fn predict(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let hidden_outputs: Vec<f64> = self.hidden_layer.iter_mut().map(|p| p.predict(inputs)).collect();
        self.output_layer.iter_mut().map(|p| p.predict(&hidden_outputs)).collect()
    }

    pub fn train(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>) {
        let hidden_outputs: Vec<f64> = self.hidden_layer.iter_mut().map(|p| p.predict(inputs)).collect();
        let final_outputs: Vec<f64> = self.output_layer.iter_mut().map(|p| p.predict(&hidden_outputs)).collect();

        let output_errors: Vec<f64> = targets.iter().zip(&final_outputs).map(|(target, output)| target - output).collect();

        for (perceptron, target) in self.output_layer.iter_mut().zip(targets) {
            perceptron.train(&hidden_outputs, *target);
        }

        let hidden_errors: Vec<f64> = self.hidden_layer.iter().enumerate().map(|(i, _)| {
            self.output_layer.iter().zip(&output_errors).map(|(out_perceptron, &error)| {
                out_perceptron.weights[i] * error
            }).sum()
        }).collect();

        for ((perceptron, &hidden_error), &hidden_output) in self.hidden_layer.iter_mut().zip(&hidden_errors).zip(&hidden_outputs) {
            perceptron.train(inputs, hidden_error + hidden_output);
        }
        
    }
}
