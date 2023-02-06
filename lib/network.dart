import 'package:neural_network/activation_functions/activation_function.dart';
import 'package:neural_network/data_point.dart';

import 'layer.dart';

class NeuralNetwork{
  List<Layer> layers = [];
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

  double cost(DataPoint point) {
    List<double> evaluatedValues = computeOutput(point.inputs);
    Layer lastLayer = layers[layers.length - 1];
    double error = 0;
    for (int i = 0; i < evaluatedValues.length; i++) {
      error += lastLayer.valueError(evaluatedValues[i], point.expectedOutputs[i]);
    }
    return error;
  }

  double costs(List<DataPoint> points) {
    double sumError = 0;
    for (DataPoint point in points) {
      sumError += cost(point);
    }
    return sumError / points.length;
  }

  int correctClassified(List<DataPoint> points) {
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
}