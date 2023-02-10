import 'dart:math';

import 'package:neural_network/activation_functions/activation_function.dart';
import 'package:neural_network/data_point.dart';

import 'layer.dart';

class NeuralNetwork {
  List<Layer> layers = [];
  double learnRate = 0.01;
  // Used for momentum
  double beta1 = 0.9;
  // Used for rmsProp
  double beta2 = 0.999;
  // To not divide by zero
  double epsilon = 1e-8;
  int timeStep = 0;

  /// Add layer to the network
  void addLayer(Layer layer) {
    layers.add(layer);
  }

  /// Compute output for the given input. It calculates output for first layer and
  /// passes result as input to the next layer and repeats that until last layer
  List<double> computeOutput(List<double> inputs) {
    List<double> outputs = inputs;
    for (Layer layer in layers) {
      outputs = layer.calculateOutput(outputs);
    }
    return outputs;
  }

  /// Change activation function for all layers to the new one
  void changeAllActivationFunctions(ActivationFunction newFunction) {
    for (Layer layer in layers) {
      layer.changeActivationFunction(newFunction);
    }
  }

  /// Return the square of difference between two points
  double getCost(double x, double y) {
    double diff = x - y;
    return diff * diff;
  }

  /// Return the derivative of difference between two points
  double getCostDerivative(double x, double y) {
    double diff = x - y;
    return 2 * diff;
  }

  /// Return the cost for single DataPoint
  double getPointCost(DataPoint point) {
    List<double> predictedOutputs = computeOutput(point.inputs);
    double error = 0;
    for (int i = 0; i < predictedOutputs.length; i++) {
      error += getCost(predictedOutputs[i], point.expectedOutputs[i]);
    }
    return error;
  }

  /// Return the sum of costs for all the DataPoints
  double getSumCosts(List<DataPoint> points) {
    double sumError = 0;
    for (DataPoint point in points) {
      sumError += getPointCost(point);
    }
    return sumError / points.length;
  }

  /// Return the amount of points that were correctly classified
  int getCorrectClassified(List<DataPoint> points) {
    int amount = 0;
    for (DataPoint point in points) {
      List<double> evaluation = computeOutput(point.inputs);

      int evaluationMaxIndex = 0;
      int expectedMaxIndex = 0;

      for (int i = 0; i < evaluation.length; i++) {
        if (evaluation[i] > evaluation[evaluationMaxIndex]) {
          evaluationMaxIndex = i;
        }
        if (point.expectedOutputs[i] > point.expectedOutputs[expectedMaxIndex]) {
          expectedMaxIndex = i;
        }
      }
      if (evaluationMaxIndex == expectedMaxIndex) {
        amount += 1;
      }
    }
    return amount;
  }

  /// Randomize every weight and bias available in the network
  void randomizeParameters() {
    Random random = Random();
    for (Layer layer in layers) {
      for (int i = 0; i < layer.weights.length; i++) {
        layer.weights[i] = random.nextDouble() * 2 - 1;
      }
      for (int i = 0; i < layer.biases.length; i++) {
        layer.biases[i] = random.nextDouble() * 2 - 1;
      }

      // Since we're changing parameters we need to reset optimizer
      layer.clearOptimizer();
      timeStep = 0;
    }
  }

  /// Return small portion of input data
  List<DataPoint> getBatchOfInput(List<DataPoint> points, int batchAmount) {
    // If batch amount is equal or larger to amount of DataPoints, then return whole list
    if (batchAmount >= points.length) {
      return points;
    }

    // Get all legal indexes for points list
    List<int> indexes = List.generate(points.length, (index) => index);

    // Shuffle it to get indexes in random order
    indexes.shuffle();

    // Collect first batchAmount of points
    List<DataPoint> batchPoints = [];
    for(int i = 0; i < batchAmount; i++) {
      DataPoint point = points.elementAt(indexes[i]);
      batchPoints.add(point);
    }
    return batchPoints;
  }

  void computeNodeCosts(DataPoint point) {
    // First we compute node costs for output layer
    Layer lastLayer = layers[layers.length - 1];

    for (int n = 0; n < lastLayer.outputAmount; n++) {
      lastLayer.nodeValues[n] =
          lastLayer.activationFunction.getActivationDerivativeOutput(
              lastLayer.equationResults[n]) * getCostDerivative(
              lastLayer.activationResults[n], point.expectedOutputs[n]);
    }

    // Then we compute node costs for all hidden layers going backwards
    for(int l = layers.length - 2; l >= 0; l--) {
      Layer hiddenLayer = layers[l];
      Layer nextLayer = layers[l + 1];

      // For each node in hidden layer calculate node value
      for (int n = 0; n < hiddenLayer.outputAmount; n++) {
        // Reset node value before computing derivative
        hiddenLayer.nodeValues[n] = 0;

        // Output of node from hidden layer affects inputs to ALL nodes in next layer,
        // so we calculate derivative of it's weight times value for that node in next layer
        for (int nNext = 0; nNext < nextLayer.outputAmount; nNext++) {
          hiddenLayer.nodeValues[n] +=
              nextLayer.weights[n + nNext * hiddenLayer.outputAmount] *
                  nextLayer.nodeValues[nNext];
        }

        // Everything we multiply by how inputs for node from hidden layer affects it's output
        hiddenLayer.nodeValues[n] *=
            hiddenLayer.activationFunction.getActivationDerivativeOutput(
                hiddenLayer.equationResults[n]);
      }
    }
  }

  /// Compute node values and gradients for every layer
  void computeGradients(DataPoint point) {
    // Loop through all layers
    for(int l = layers.length - 1; l >= 0; l--) {
      Layer layer = layers[l];

      // Default to point inputs if this was last layer
      List<double> activationsFromLayerBefore = point.inputs;

      // If there is layer before this one, then get activations from there
      if (l - 1 >= 0) {
        activationsFromLayerBefore = layers[l - 1].activationResults;
      }

      // Compute gradient for each weights from this layer
      // Derivative of equation by given weight is input from the previous layer
      // which is output of activation function from layer before this
      for (int n = 0; n < layer.outputAmount; n++) {
        for (int i = 0; i < layer.inputAmount; i++) {
          layer.gradientWeights[i + n * layer.inputAmount] += activationsFromLayerBefore[i] * layer.nodeValues[n];
        }
      }

      // Compute gradient for each bias from this layer
      // Derivative of equation by given bias is just 1
      for (int n = 0; n < layer.outputAmount; n++) {
        layer.gradientBias[n] += 1 * layer.nodeValues[n];
      }
    }
  }

  /// Apply all stored gradients with dividing gradients by amount of points
  /// that was used to create this gradient
  void applyAllGradients(int pointsAmount) {
    // Average weight gradients
    for (Layer layer in layers) {
      for (int w = 0; w < layer.weights.length; w++) {
        layer.gradientWeights[w] /= pointsAmount;
      }
    }

    // Average bias gradients
    for (Layer layer in layers) {
      for (int b = 0; b < layer.biases.length; b++) {
        layer.gradientBias[b] /= pointsAmount;
      }
    }

    // Update weights using ADAM optimizer
    for (Layer layer in layers) {
      for (int w = 0; w < layer.weights.length; w++) {
        layer.momentumWeights[w] = beta1 * layer.momentumWeights[w] + (1 - beta1) * layer.gradientWeights[w];
        layer.rmsWeights[w] = beta2 * layer.rmsWeights[w] + (1 - beta2) * pow(layer.gradientWeights[w], 2);
        double moment = layer.momentumWeights[w] / (1 - pow(beta1, timeStep));
        double rms = layer.rmsWeights[w] / (1 - pow(beta2, timeStep));
        layer.weights[w] -= learnRate * moment / (sqrt(rms) + epsilon);
      }
    }

    // Update biases using ADAM optimizer
    for (Layer layer in layers) {
      for (int b = 0; b < layer.biases.length; b++) {
        layer.momentumBias[b] = beta1 * layer.momentumBias[b] + (1 - beta1) * layer.gradientBias[b];
        layer.rmsBias[b] = beta2 * layer.rmsBias[b] + (1 - beta2) * pow(layer.gradientBias[b], 2);
        double moment = layer.momentumBias[b] / (1 - pow(beta1, timeStep));
        double rms = layer.rmsBias[b] / (1 - pow(beta2, timeStep));
        layer.biases[b] -= learnRate * moment / (sqrt(rms) + epsilon);
      }
    }
  }

  /// Clear any stored gradients from layers for next gradient operation
  void clearGradients() {
    for (Layer layer in layers) {
      layer.clearGradientValues();
    }
  }

  /// Run gradient descent algorithm
  void runGradientDescent(List<DataPoint> points, int batchAmount) {
    List<DataPoint> miniBatch = getBatchOfInput(points, batchAmount);

    // We update time step to indicate new iteration
    timeStep += 1;

    for (DataPoint point in miniBatch) {
      // Run compute output for single point to get activation and equation results filled
      computeOutput(point.inputs);

      // Compute the node values
      computeNodeCosts(point);

      // Compute gradients for every layer
      computeGradients(point);
    }
    // Finally apply all collected gradients
    applyAllGradients(points.length);

    // At last clear gradients for future gradient descent
    clearGradients();
  }
}