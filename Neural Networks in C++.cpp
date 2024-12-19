#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>  // For controlling output formatting

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

int main() {
    // Training data for XOR
    double inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    double expected_output[4] = { 0, 1, 1, 0 };

    // Initialize weights and biases
    double input_weights[2][3];  // Weights for input to hidden layer (3 hidden neurons)
    double hidden_weights[3];    // Weights for hidden to output layer
    double hidden_bias[3];       // Biases for hidden layer
    double output_bias;          // Bias for output layer

    // Random initialization of weights and biases
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            input_weights[i][j] = (double)rand() / RAND_MAX;  // Random between 0 and 1
        }
    }
    for (int j = 0; j < 3; j++) {
        hidden_bias[j] = (double)rand() / RAND_MAX;
        hidden_weights[j] = (double)rand() / RAND_MAX;
    }
    output_bias = (double)rand() / RAND_MAX;

    // Learning rate
    double lr = 0.1;

    // Training the network
    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 4; i++) {
            // Forward propagation
            double input1 = inputs[i][0];
            double input2 = inputs[i][1];

            // Hidden layer
            double hidden_input[3];
            double hidden_output[3];
            for (int j = 0; j < 3; j++) {
                hidden_input[j] = input1 * input_weights[0][j] + input2 * input_weights[1][j] + hidden_bias[j];
                hidden_output[j] = sigmoid(hidden_input[j]);
            }

            // Output layer
            double output_input = 0;
            for (int j = 0; j < 3; j++) {
                output_input += hidden_output[j] * hidden_weights[j];
            }
            output_input += output_bias;
            double output = sigmoid(output_input);

            // Backpropagation
            double error = expected_output[i] - output;
            double output_delta = error * sigmoid_derivative(output);

            // Update weights and biases for the output layer
            for (int j = 0; j < 3; j++) {
                hidden_weights[j] += lr * output_delta * hidden_output[j];
            }
            output_bias += lr * output_delta;

            // Calculate deltas for the hidden layer
            double hidden_delta[3];
            for (int j = 0; j < 3; j++) {
                hidden_delta[j] = output_delta * hidden_weights[j] * sigmoid_derivative(hidden_output[j]);
            }

            // Update weights and biases for the hidden layer
            for (int j = 0; j < 3; j++) {
                input_weights[0][j] += lr * hidden_delta[j] * input1;
                input_weights[1][j] += lr * hidden_delta[j] * input2;
                hidden_bias[j] += lr * hidden_delta[j];
            }
        }
    }

    // Test the trained network
    std::cout << "XOR Output after training:\n";
    for (int i = 0; i < 4; i++) {
        double input1 = inputs[i][0];
        double input2 = inputs[i][1];

        // Hidden layer
        double hidden_input[3];
        double hidden_output[3];
        for (int j = 0; j < 3; j++) {
            hidden_input[j] = input1 * input_weights[0][j] + input2 * input_weights[1][j] + hidden_bias[j];
            hidden_output[j] = sigmoid(hidden_input[j]);
        }

        // Output layer
        double output_input = 0;
        for (int j = 0; j < 3; j++) {
            output_input += hidden_output[j] * hidden_weights[j];
        }
        output_input += output_bias;
        double output = sigmoid(output_input);

        std::cout << input1 << " XOR " << input2 << " = " << round(output) << std::endl;
    }

    // Print the final weights and biases
    std::cout << "\nFinal Weights and Biases after Training:\n";
    
    // Print input to hidden weights
    std::cout << "Input to Hidden Weights:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << "Weight[" << i << "][" << j << "] = " << std::setprecision(4) << input_weights[i][j] << "\n";
        }
    }

    // Print hidden to output weights
    std::cout << "\nHidden to Output Weights:\n";
    for (int j = 0; j < 3; j++) {
        std::cout << "Hidden Weight[" << j << "] = " << std::setprecision(4) << hidden_weights[j] << "\n";
    }

    // Print hidden layer biases
    std::cout << "\nHidden Biases:\n";
    for (int j = 0; j < 3; j++) {
        std::cout << "Hidden Bias[" << j << "] = " << std::setprecision(4) << hidden_bias[j] << "\n";
    }

    // Print output bias
    std::cout << "\nOutput Bias: " << std::setprecision(4) << output_bias << "\n";

    return 0;
}