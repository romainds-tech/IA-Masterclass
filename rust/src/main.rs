#![allow(warnings)]

use std::time::Instant;
mod perceptron;
mod layer;
use crate::perceptron::ActivationType;
use crate::layer::Layer;


fn main() {
    let start = Instant::now();

    let dataset: Vec<([f64; 3], i32)> = vec![
        ([0.12, 0.90, 0.10], 1),
        ([0.10, 0.70, 0.40], 1),
        ([0.14, 0.40, 0.20], 1),
        ([0.01, 0.60, 0.60], 1),
        ([0.08, 0.50, 0.30], 1),
        ([0.25, 0.10, 0.10], 0),
        ([0.30, 0.15, 0.10], 0),
        ([0.28, 0.05, 0.30], 0),
        ([0.21, 0.0, 0.0], 0),
        ([0.40, 0.10, 0.10], 0),
    ];

    let epoch = 100;
    let mut all_errors = Vec::new();

    for _ in 0..500 {
        let mut layer = Layer::new(
            3,
            (9, ActivationType::Relu),
            (1, ActivationType::Threshold),
        );

        for _ in 0..epoch {
            for (inputs, target) in &dataset {
                layer.train(&inputs.to_vec(), &vec![*target as f64]);
            }
        }

        let mut error_mean = 0.0;
        for (inputs, target) in &dataset {
            error_mean += (*target as f64 - layer.predict(&inputs.to_vec())[0]).abs();
        }
        error_mean /= dataset.len() as f64;

        all_errors.push(error_mean);
    }

    let error_avg: f64 = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    println!("Moyenne des erreurs: {}", error_avg);

    println!("Temps écoulé: {} secondes", start.elapsed().as_secs_f64());
}
