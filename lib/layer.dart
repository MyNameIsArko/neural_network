import 'package:neural_network/activation_functions/activation_function.dart';

class Layer {
  late List<double> weights;
  late List<double> biases;
  int inputAmount;
  int outputAmount;
  ActivationFunction activationFunction;
  late List<double> equationResults;
  late List<double> activationResults;
  late List<double> gradientWeights;
  late List<double> gradientBias;
  late List<double> nodeValues;
  late List<double> momentumWeights;
  late List<double> momentumBias;
  late List<double> rmsWeights;
  late List<double> rmsBias;

  Layer(this.inputAmount, this.outputAmount, this.activationFunction) {
    weights = List.filled(inputAmount * outputAmount, 0.5);
    biases = List.filled(outputAmount, 0.5);
    equationResults = List.filled(outputAmount, 0);
    activationResults = List.filled(outputAmount, 0);
    gradientWeights = List.filled(inputAmount * outputAmount, 0);
    gradientBias = List.filled(outputAmount, 0);
    nodeValues = List.filled(outputAmount, 0);
    momentumWeights = List.filled(inputAmount * outputAmount, 0);
    momentumBias = List.filled(outputAmount, 0);
    rmsWeights = List.filled(inputAmount * outputAmount, 0);
    rmsBias = List.filled(outputAmount, 0);
  }

  // Calculate output from given inputs, by multiplying inputs by weights
  List<double> calculateOutput(List<double> inputs) {
    if (inputs.length != inputAmount) {
      throw FormatException("Inputs given (${inputs.length}) and expected amount ($inputAmount) does not match!");
    }
    List<double> outputs = List.filled(outputAmount, 0);
    // For each node in layer calculate it's weights with inputs
    for (int i = 0; i < outputAmount; i++) {
      for (int j = 0; j < inputAmount; j++) {
        outputs[i] += weights[j + i * inputAmount] * inputs[j];
      }
      // Add bias to the equation
      outputs[i] += biases[i];
      // Save equation for backpropagation use
      equationResults[i] = outputs[i];
      // Change the output to fire value from activation function
      outputs[i] = activationFunction.getActivationOutput(outputs[i]);
      // Save activation for backpropagation use
      activationResults[i] = outputs[i];
    }
    return outputs;
  }

  void updateWeight(int index, double value) {
    if (index < weights.length) {
      weights[index] = value;
    }
  }

  void updateBias(int index, double value) {
    if (index < biases.length) {
      biases[index] = value;
    }
  }

  void changeActivationFunction(ActivationFunction newFunction) {
    activationFunction = newFunction;
  }

  void clearGradientValues() {
    gradientWeights = List.filled(inputAmount * outputAmount, 0);
    gradientBias = List.filled(outputAmount, 0);
    nodeValues = List.filled(outputAmount, 0);
  }

  void clearOptimizer() {
    momentumWeights = List.filled(inputAmount * outputAmount, 0);
    momentumBias = List.filled(outputAmount, 0);
    rmsWeights = List.filled(inputAmount * outputAmount, 0);
    rmsBias = List.filled(outputAmount, 0);
  }
}