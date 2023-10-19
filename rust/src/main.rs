extern crate rand;
mod perceptron;
mod dense;
use perceptron::{ActivationType};
use dense::{Dense};

fn main() {
    /*  This is a simple dataset to test the perceptron
        First value is the temperature, the seconde is the humidity and the last one
        is the wind speed
        The expected value is 1.0 if it's good condition to wearing a jacket, 0.0
        otherwise
    */

    let dataset: [([f64; 3], f64); 10] = [
        ([2.0, 60.0, 20.0], 1.0),
        ([2.0, 70.0, 10.0], 1.0),
        ([2.0, 80.0, 15.0], 1.0),
        ([12.0, 50.0, 35.0], 1.0),
        ([12.0, 30.0, 80.0], 1.0),
        //
        ([40.0, 10.0, 20.0], 0.0),
        ([35.0, 10.0, 0.0], 0.0),
        ([28.0, 5.0, 20.0], 0.0),
        ([32.0, 10.0, 5.0], 0.0),
        ([30.0, 15.0, 2.0], 0.0),
    ];

    // Création d'une instance de Dense
    let mut dense = Dense::new(3, 2, ActivationType::RELU, 2, ActivationType::SIGMOID);

    // Faire une prédiction
    let pred = dense.forward(vec![1.0, 2.0, 3.0]);

    // Print la prédiction
    println!("{:?}", pred);
}
