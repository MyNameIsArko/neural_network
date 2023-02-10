import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';
import 'package:neural_network/activation_functions/lrelu_function.dart';
import 'package:neural_network/data_point.dart';
import 'package:neural_network/layer.dart';
import 'package:neural_network/network.dart';

void main() async {
  Stream<List<int>> input = File("lib/problems/mnist_train.csv").openRead();
  List<List<dynamic>> fields = await input.transform(utf8.decoder).transform(const CsvToListConverter(eol: "\n")).toList();

  List<DataPoint> dataPoints = [];
  for(List<dynamic> field in fields) {
    List<double> numbers = field.sublist(1).map((item) => (item as int).toDouble()).toList();
    DataPoint point = DataPoint(numbers);
    List<double> outputs = List.filled(10, 0);
    int index = field[0] as int;
    outputs[index] = 1;
    point.expectedOutputs = outputs;

    dataPoints.add(point);
  }

  NeuralNetwork network = NeuralNetwork();
  network.addLayer(Layer(784, 20, LReLUFunction()));
  network.addLayer(Layer(20, 20, LReLUFunction()));
  network.addLayer(Layer(20, 10, LReLUFunction()));

  network.randomizeParameters();

  print(network.getSumCosts(dataPoints));

  dataPoints.shuffle();

  List<DataPoint> learningData = dataPoints.sublist(80);
  List<DataPoint> testData = dataPoints.sublist(0, 80);

  for (int i = 0; i < 1000; i++) {
      print("---------------------------");
      print("Epoch: $i");
      print("Learning cost: ${network.getSumCosts(learningData)}");
      print("Test cost: ${network.getSumCosts(testData)}");
    network.runGradientDescent(learningData, 40);
  }
}