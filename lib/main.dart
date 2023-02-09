import 'dart:ui';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:neural_network/activation_functions/hyperbolic_tangent_function.dart';
import 'package:neural_network/activation_functions/linear_function.dart';
import 'package:neural_network/activation_functions/lrelu_function.dart';
import 'package:neural_network/activation_functions/sigmoid_function.dart';
import 'package:neural_network/activation_functions/threshold_function.dart';
import 'package:neural_network/data_point.dart';
import 'package:neural_network/data_spread/corner_parabola_2d.dart';
import 'package:neural_network/network.dart';
import 'package:neural_network/precision.dart';

import 'activation_functions/activation_function.dart';
import 'activation_functions/elu_function.dart';
import 'activation_functions/relu_function.dart';
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
  List<DataPoint> dataPoints = [];
  NeuralNetwork network = NeuralNetwork();
  List<ActivationFunction> activationFunctions = [
    LinearFunction(),
    ReLUFunction(),
    LReLUFunction(),
    ELUFunction(),
    ThresholdFunction(),
    SigmoidFunction(),
    HyperbolicTangentFunction(),
  ];
  List<Precision> gridPrecision = [
    Precision(4, "Ultra Performance"),
    Precision(2, "Performance"),
    Precision(1, "Quality"),
    Precision(0.5, "Ultra Quality")
  ];
  late Precision selectedPrecision;
  String selectedFunctionName = LinearFunction().getName();
  @override
  void initState() {
    selectedPrecision = gridPrecision.first;
    for (double i = -40; i < 41; i+=6) {
      for (double j = -40; j < 41; j+=12) {
        dataPoints.add(DataPoint([i, j], CornerParabola2D()));
      }
    }
    Layer hiddenLayer1 = Layer(2, 3, LinearFunction());
    // Layer hiddenLayer2 = Layer(3, 3, LinearFunction());
    Layer outputLayer = Layer(3, 2, LinearFunction());
    network.addLayer(hiddenLayer1);
    // network.addLayer(hiddenLayer2);
    network.addLayer(outputLayer);
    super.initState();
  }

  Color getColor(double x1, double x2) {
    double o1 = network.computeOutput([x1, x2])[0];
    double o2 = network.computeOutput([x1, x2])[1];
    if (o1 > o2) {
      return Colors.blue;
    } else {
      return Colors.red;
    }
  }

  List<ScatterSpot> getScatterSpots() {
    List<ScatterSpot> spots = [];
    for (double i = -40; i < 41; i += selectedPrecision.value) {
      for (double j = -40; j < 41; j += selectedPrecision.value) {
        spots.add(ScatterSpot(i, j, color: getColor(i, j), radius: 8 * selectedPrecision.value));
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
              // Disabled because of slider min/max value interrupting values of learning

              // Column(
              //   children: <Widget>[
              //     const Text(
              //       "Weights:",
              //       style: TextStyle(
              //         fontSize: 20,
              //       ),
              //     )
              //   ] + List<Widget>.generate(network.weightSize(), (index) {
              //     Layer layer;
              //     int j = 0;
              //     if (index < 6) {
              //       layer = network.layers[0];
              //       j = index;
              //     } else {
              //       layer = network.layers[1];
              //       j = index - 6;
              //     }
              //     return SizedBox(
              //       height: 20,
              //       width: 300,
              //       child: SliderTheme(
              //         data: const SliderThemeData(
              //           thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
              //         ),
              //         child: Slider(
              //           value: layer.weights[j],
              //           max: 20,
              //           min: -20,
              //           onChanged: (value) {
              //             setState(() {
              //               layer.updateWeight(j, value);
              //             });
              //           },
              //         ),
              //       ),
              //     );
              //   })
              // ),
              // Column(
              //   children: <Widget>[
              //     const Text(
              //       "Biases:",
              //       style: TextStyle(
              //         fontSize: 20,
              //       ),
              //     )
              //   ] + List<Widget>.generate(network.biasSize(), (index) {
              //     Layer layer;
              //     int j = 0;
              //     if (index < 3) {
              //       layer = network.layers[0];
              //       j = index;
              //     } else {
              //       layer = network.layers[1];
              //       j = index - 3;
              //     }
              //     return SizedBox(
              //       height: 20,
              //       width: 300,
              //       child: SliderTheme(
              //         data: const SliderThemeData(
              //           thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
              //         ),
              //         child: Slider(
              //           value: layer.biases[j],
              //           max: 20,
              //           min: -20,
              //           onChanged: (value) {
              //             setState(() {
              //               layer.updateBias(j, value);
              //             });
              //           },
              //         ),
              //       ),
              //     );
              //   })
              // ),
              Spacer(),
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
                  ),
                  Image.asset(
                    "functions_images/$selectedFunctionName.png",
                    width: 200,
                    height: 200,
                  ),
                ]
              ),
              const SizedBox(
                width: 25,
              ),
              Column(
                children: [
                  const Text(
                    "Grid accuracy:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  ),
                  DropdownButton(
                    value: selectedPrecision,
                    onChanged: (Precision? value) {
                      setState(() {
                        selectedPrecision = value!;
                      });
                    },
                    items: gridPrecision.map<DropdownMenuItem<Precision>>((Precision precision) {
                      return DropdownMenuItem(
                          value: precision,
                          child: Text(precision.name)
                      );
                    }).toList(),
                  ),
                  const SizedBox(
                    height: 25,
                  ),
                  const Text(
                    "Cost:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  ),
                  Text(network.getSumCosts(dataPoints).toStringAsFixed(2)),
                  const SizedBox(
                    height: 25,
                  ),
                  const Text(
                    "Correct:",
                    style: TextStyle(
                      fontSize: 20,
                    ),
                  ),
                  Text("${network.getCorrectClassified(dataPoints)} / ${dataPoints.length}"),
                ],
              ),
              const SizedBox(
                width: 25,
              ),
              Column(
                children: [
                  TextButton(
                    onPressed: () {
                      setState(() {
                        network.randomizeParameters();
                      });
                    },
                    style: ButtonStyle(backgroundColor: MaterialStateProperty.all(Colors.grey.shade200)),
                    child:  const Text("Randomize weights and biases"),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  TextButton(
                    onPressed: () {
                      setState(() {
                        network.runGradientDescent(network.getBatchOfInput(dataPoints, 10));
                      });
                      // print("================================================");
                      // print("Weights:");
                      // for (Layer layer in network.layers) {
                      //   print(layer.weights);
                      // }
                      // print("Biases:");
                      // for (Layer layer in network.layers) {
                      //   print(layer.weights);
                      // }
                      // print("================================================");
                    },
                    style: ButtonStyle(backgroundColor: MaterialStateProperty.all(Colors.grey.shade200)),
                    child:  const Text("Learn one step"),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  TextButton(
                    onPressed: () {
                        int i = 0;
                        // One bad classified point is "good enough"
                        while (network.getCorrectClassified(dataPoints) < dataPoints.length - 1 && i < 15e4) {
                          network.runGradientDescent(network.getBatchOfInput(dataPoints, 10));
                          if (i % 1000 == 0) {
                            setState(() {
                              print(i);
                            });
                          }
                          i += 1;
                        }
                    },
                    style: ButtonStyle(backgroundColor: MaterialStateProperty.all(Colors.grey.shade200)),
                    child:  const Text("Learn a good chunk"),
                  ),
                ],
              ),
              Spacer(),
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
                      scatterSpots: dataPoints.map((DataPoint point) {
                        return ScatterSpot(point.inputs[0],point.inputs[1], color: point.pointColor);
                      }).toList(),
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
