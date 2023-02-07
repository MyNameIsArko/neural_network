import 'dart:math';

import 'package:neural_network/activation_functions/activation_function.dart';
import 'package:neural_network/data_point.dart';

import 'layer.dart';

class NeuralNetwork{
  List<Layer> layers = [];
  double learnRate = 0.1;
  void addLayer(Layer layer) {
    layers.add(layer);
  }

  List<double> computeOutput(List<double> inputs) {
    List<double> outputs = inputs;
    for (Layer layer in layers) {
      outputs = layer.calculateOutput(outputs);
    }
    return outputs;
  }

  void changeAllActivationFunctions(ActivationFunction newFunction) {
    for (Layer layer in layers) {
      layer.changeActivationFunction(newFunction);
    }
  }

  int weightSize() {
    int size = 0;
    for (Layer layer in layers) {
      size += layer.weights.length;
    }
    return size;
  }

  int biasSize() {
    int size = 0;
    for (Layer layer in layers) {
      size += layer.biases.length;
    }
    return size;
  }

  double getCost(double x, double y) {
    double diff = x - y;
    return diff * diff;
  }

  double getCostDerivative(double x, double y) {
    double diff = x - y;
    return 2 * diff;
  }

  double getPointCost(DataPoint point) {
    List<double> evaluatedValues = computeOutput(point.inputs);
    double error = 0;
    for (int i = 0; i < evaluatedValues.length; i++) {
      error += getCost(evaluatedValues[i], point.expectedOutputs[i]);
    }
    return error;
  }

  double getSumCosts(List<DataPoint> points) {
    double sumError = 0;
    for (DataPoint point in points) {
      sumError += getPointCost(point);
    }
    return sumError / points.length;
  }

  int getCorrectClassified(List<DataPoint> points) {
    int amount = 0;
    for (DataPoint point in points) {
      List<double> evaluation = computeOutput(point.inputs);
      if (evaluation[0] > evaluation[1] && point.expectedOutputs[0] > point.expectedOutputs[1]) {
        amount += 1;
      } else if (evaluation[0] <= evaluation[1] && point.expectedOutputs[0] <= point.expectedOutputs[1]) {
        amount += 1;
      }
    }
    return amount;
  }

  List<double> getGradient(List<DataPoint> inputs) {
    List<double> gradient = List.filled(weightSize() + biasSize(), 0);

    double h = 0.0000001;

    int i = 0;

    double cost1 = getSumCosts(inputs);

    // Compute weights gradient
    for (Layer layer in layers) {
      for (int w = 0; w < layer.weights.length; w++) {
        layer.weights[w] += h;
        double cost2 = getSumCosts(inputs);
        layer.weights[w] -= h;
        gradient[i] = (cost2 - cost1) / h;
        i += 1;
      }
    }

    // Compute biases gradient
    for (Layer layer in layers) {
      for (int b = 0; b < layer.biases.length; b++) {
        layer.biases[b] += h;
        double cost2 = getSumCosts(inputs);
        layer.biases[b] -= h;
        gradient[i] = (cost2 - cost1) / h;
        i += 1;
      }
    }

    return gradient;
  }

  void runGradientDescent(List<DataPoint> inputs) {
    List<double> gradient = getGradient(inputs);

    int i = 0;

    // Update weights
    for (Layer layer in layers) {
      for (int w = 0; w < layer.weights.length; w++) {
        layer.weights[w] -= learnRate * gradient[i];
        i += 1;
      }
    }

    // Update biases
    for (Layer layer in layers) {
      for (int b = 0; b < layer.biases.length; b++) {
        layer.biases[b] -= learnRate * gradient[i];
        i += 1;
      }
    }
  }

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

  List<DataPoint> getBatchOfInput(List<DataPoint> points) {
    List<int> indexes = List.generate(points.length, (index) => index);
    indexes.shuffle();

    List<DataPoint> batchPoints = [];
    for(int i = 0; i < 10; i++) {
      DataPoint point = points.elementAt(indexes[i]);
      batchPoints.add(point);
    }
    return batchPoints;
  }

  // double nodeValue(Layer layer, DataPoint point) {
  //   return layer.activationFunction.getActivationDerivativeOutput(point.inputs[0]) * getCostDerivative(point.inputs[0], point.expectedOutputs[0]);
  // }
  //
  // void updateLastLayer(List<DataPoint> points) {
  //   Layer lastLayer = layers[layers.length - 1];
  //   List<double> gradient = List.filled(lastLayer.weights.length + lastLayer.biases.length, 0);
  //
  //
  // }
}