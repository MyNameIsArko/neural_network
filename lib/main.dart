import 'dart:ui';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:neural_network/activation_functions/hyperbolic_tangent_function.dart';
import 'package:neural_network/activation_functions/linear_function.dart';
import 'package:neural_network/activation_functions/relu_function.dart';
import 'package:neural_network/activation_functions/sigmoid_function.dart';
import 'package:neural_network/activation_functions/threshold_function.dart';
import 'package:neural_network/network.dart';

import 'activation_functions/activation_function.dart';
import 'layer.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Neural Network',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final List<double> points = List<double>.generate(100, (index) => index - 50);
  List<ScatterSpot> scatterPoints = [];
  NeuralNetwork network = NeuralNetwork();
  List<ActivationFunction> activationFunctions = [
    LinearFunction(),
    ReLUFunction(),
    ThresholdFunction(),
    SigmoidFunction(),
    HyperbolicTangentFunction(),
  ];
  String selectedFunctionName = LinearFunction().getName();
  @override
  void initState() {
    for (int i = -40; i < 41; i+=4) {
      for (int j = -40; j < 41; j+=8) {
        Color pointColor = Colors.red;
        if (j < -i - 10 && j < 10 && i < 10) {
          pointColor = Colors.blue;
        }
        scatterPoints.add(ScatterSpot(i.toDouble(), j.toDouble(), color: pointColor));
      }
    }
    Layer hiddenLayer = Layer(2, 3, LinearFunction());
    Layer outputLayer = Layer(3, 1, LinearFunction());
    network.addLayer(hiddenLayer);
    network.addLayer(outputLayer);
    super.initState();
  }

  Color getColor(double x1, double x2) {
    double o = network.computeOutput([x1, x2])[0];
    if (o > 0) {
      return Colors.blue;
    } else {
      return Colors.red;
    }
  }

  List<ScatterSpot> getScatterSpots() {
    List<ScatterSpot> spots = [];
    for (double i = -40; i < 41; i += 2) {
      for (double j = -40; j < 41; j += 2) {
        spots.add(ScatterSpot(i, j, color: getColor(i, j), radius: 16));
      }
    }
    return spots;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: <Widget>[
          Row(
            children: [
              Column(
                children: <Widget>[
                  const Text(
                    "Weights:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  )
                ] + List<Widget>.generate(9, (index) {
                  Layer layer;
                  int j = 0;
                  if (index < 6) {
                    layer = network.layers[0];
                    j = index;
                  } else {
                    layer = network.layers[1];
                    j = index - 6;
                  }
                  return SizedBox(
                    height: 20,
                    width: 300,
                    child: SliderTheme(
                      data: const SliderThemeData(
                        thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
                      ),
                      child: Slider(
                        value: layer.weights[j],
                        max: 10,
                        min: -10,
                        onChanged: (value) {
                          setState(() {
                            layer.updateWeight(j, value);
                          });
                        },
                      ),
                    ),
                  );
                })
              ),
              Column(
                children: <Widget>[
                  const Text(
                    "Biases:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  )
                ] + List<Widget>.generate(4, (index) {
                  Layer layer;
                  int j = 0;
                  if (index < 3) {
                    layer = network.layers[0];
                    j = index;
                  } else {
                    layer = network.layers[1];
                    j = index - 3;
                  }
                  return SizedBox(
                    height: 20,
                    width: 300,
                    child: SliderTheme(
                      data: const SliderThemeData(
                        thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
                      ),
                      child: Slider(
                        value: layer.biases[j],
                        max: 100,
                        min: -100,
                        onChanged: (value) {
                          setState(() {
                            layer.updateBias(j, value);
                          });
                        },
                      ),
                    ),
                  );
                })
              ),
              Column(
                children: <Widget>[
                  const Text(
                    "Activation Function:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  ),
                  DropdownButton(
                    value: selectedFunctionName,
                    onChanged: (String? value) {
                      setState(() {
                        ActivationFunction newFunction = LinearFunction();
                        for (ActivationFunction function in activationFunctions) {
                          if (function.getName() == value!) {
                            newFunction = function;
                            break;
                          }
                        }
                        selectedFunctionName = value!;
                        network.changeAllActivationFunctions(newFunction);
                      });
                    },
                    items: activationFunctions.map<DropdownMenuItem<String>>((ActivationFunction function) {
                      return DropdownMenuItem(
                        value: function.getName(),
                        child: Text(function.getName())
                      );
                    }).toList(),
                  )
                ]
              ),
            ],
          ),
          Flexible(
            child: Stack(
              children: [
                Opacity(
                  opacity: 0.5,
                  child: ScatterChart(
                    ScatterChartData(
                      gridData: FlGridData(show: false),
                      clipData: FlClipData.all(),
                      scatterTouchData: ScatterTouchData(enabled: false),
                      minX: -40,
                      maxX: 40,
                      minY: -40,
                      maxY: 40,
                      scatterSpots: getScatterSpots(),
                    ),
                    swapAnimationDuration: Duration(),
                  ),
                ),
                Container(
                  margin: const EdgeInsets.symmetric(vertical: 30, horizontal: 45),
                  child: ScatterChart(
                    ScatterChartData(
                      scatterSpots: scatterPoints,
                      gridData: FlGridData(show: false),
                      borderData: FlBorderData(show: false),
                      clipData: FlClipData.all(),
                      scatterTouchData: ScatterTouchData(enabled: false),
                      titlesData: FlTitlesData(show: false),
                      minX: -40,
                      maxX: 40,
                      minY: -40,
                      maxY: 40,
                    )
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }
}
