import 'package:neural_network/activation_functions/activation_function.dart';

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
}