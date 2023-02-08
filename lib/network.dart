import 'dart:math';

import 'package:neural_network/activation_functions/activation_function.dart';
import 'package:neural_network/data_point.dart';

import 'layer.dart';

class NeuralNetwork {
  List<Layer> layers = [];
  double learnRate = 0.1;

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

  /// Compute node values and gradients for output layer
  void computeGradientForLastLayer(DataPoint point) {
    Layer lastLayer = layers[layers.length - 1];

    // For each node in last layer compute node value for future usage
    for (int n = 0; n < lastLayer.outputAmount; n++) {
      lastLayer.nodeValues[n] =
          lastLayer.activationFunction.getActivationDerivativeOutput(
              lastLayer.equationResults[n]) * getCostDerivative(
              lastLayer.activationResults[n], point.expectedOutputs[n]);
    }

    // Default to point inputs if there was only last layer
    List<double> activationsFromLayerBefore = point.inputs;

    // If there is layer before last layer, then get activations from there
    if (layers.length > 1) {
      activationsFromLayerBefore = layers[layers.length - 2].activationResults;
    }

    // Compute gradient for each weights from last layer
    // Derivative of equation by given weight is input from the previous layer
    // which is output of activation function from layer before this
    for (int n = 0; n < lastLayer.outputAmount; n++) {
      for (int i = 0; i < lastLayer.inputAmount; i++) {
        lastLayer.gradientWeights[i + n * lastLayer.inputAmount] += activationsFromLayerBefore[i] * lastLayer.nodeValues[n];
      }
    }

    // Compute gradient for each bias from this layer
    // Derivative of equation by given bias is just 1
    for (int n = 0; n < lastLayer.outputAmount; n++) {
      lastLayer.gradientBias[n] += 1 * lastLayer.nodeValues[n];
    }
  }

  /// Compute node values and gradients for every layer
  void computeGradients(DataPoint point) {
    // First we need to get gradient for last layer to be able to go back through layers
    computeGradientForLastLayer(point);

    // If there isn't any hidden layers then we have ended
    if (layers.length < 2) {
      return;
    }

    // Loop through all hidden layers
    // We know that if we go backwards, layer to the right will have calculated it nodes values
    // that contains chain rule of derivatives up to the cost value
    for(int l = layers.length - 2; l >= 0; l--) {
      Layer hiddenLayer = layers[l];
      Layer nextLayer = layers[l + 1];

      // Default to point inputs if this was last layer
      List<double> activationsFromLayerBefore = point.inputs;

      // If there is layer before this one, then get activations from there
      if (l - 1 >= 0) {
        activationsFromLayerBefore = layers[layers.length - 2].activationResults;
      }

      // For each node in hidden layer calculate node value
      for (int n = 0; n < hiddenLayer.outputAmount; n++) {
        // Reset node value before computing derivative
        hiddenLayer.nodeValues[n] = 0;

        // Output of node from hidden layer affects inputs to ALL nodes in next layer,
        // so we calculate derivative of it's weight times value for that node in next layer
        for (int nNext = 0; nNext < nextLayer.outputAmount; nNext++) {
          hiddenLayer.nodeValues[n] += nextLayer.weights[n + nNext * hiddenLayer.outputAmount] * nextLayer.nodeValues[nNext];
        }

        // Everything we multiply by how inputs for node from hidden layer affects it's output
        hiddenLayer.nodeValues[n] *= hiddenLayer.activationFunction.getActivationDerivativeOutput(hiddenLayer.equationResults[n]);
      }

      // Compute gradient for each weights from this layer
      // Derivative of equation by given weight is input from the previous layer
      // which is output of activation function from layer before this
      for (int n = 0; n < hiddenLayer.outputAmount; n++) {
        for (int i = 0; i < hiddenLayer.inputAmount; i++) {
          hiddenLayer.gradientWeights[i + n * hiddenLayer.inputAmount] += activationsFromLayerBefore[i] * hiddenLayer.nodeValues[n];
        }
      }

      // Compute gradient for each bias from this layer
      // Derivative of equation by given bias is just 1
      for (int n = 0; n < hiddenLayer.outputAmount; n++) {
        hiddenLayer.gradientBias[n] += 1 * hiddenLayer.nodeValues[n];
      }
    }
  }

  /// Apply all stored gradients with dividing gradients by amount of points
  /// that was used to create this gradient
  void applyAllGradients(int pointsAmount) {
    // Update weights
    for (Layer layer in layers) {
      for (int w = 0; w < layer.weights.length; w++) {
        layer.weights[w] -= learnRate * layer.gradientWeights[w] / pointsAmount;
      }
    }

    // Update biases
    for (Layer layer in layers) {
      for (int b = 0; b < layer.biases.length; b++) {
        layer.biases[b] -= learnRate * layer.gradientBias[b] / pointsAmount;
      }
    }
  }

  /// Clear any stored gradients from layers for next gradient operation
  void clearGradients() {
    for (Layer layer in layers) {
      layer.clearGradientValues();
    }
  }

  /// Run gradient descent algorithm on a given batch of points
  void runGradientDescent(List<DataPoint> points) {
    for (DataPoint point in points) {
      // Run compute output for single point to get activation and equation results filled
      computeOutput(point.inputs);

      // Then compute node values and gradients for every layer
      computeGradients(point);
    }
    // Finally apply all collected gradients
    applyAllGradients(points.length);

    // At last clear gradients for future gradient descent
    clearGradients();
  }
}