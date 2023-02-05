import 'dart:ui';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:neural_network/network.dart';

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
  double _w1 = 1;
  double _w2 = 1;
  double _b = 0;
  List<ScatterSpot> scatterPoints = [];
  NeuralNetwork network = NeuralNetwork();
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
    Layer hiddenLayer = Layer(2, 3);
    Layer outputLayer = Layer(3, 1);
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
      body: Center(
        child: Column(
          children: [

            Row(
              children: [
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 30),
                  child: const Text("w_1:"),
                ),
                Flexible(
                  child: Slider(
                    value: _w1,
                    max: 10,
                    min: -10,
                    onChanged: (value) {
                      setState(() {
                        _w1 = value;
                      });
                    },
                  ),
                ),
              ],
            ),
            Row(
              children: [
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 30),
                  child: const Text("w_2:"),
                ),
                Flexible(
                  child: Slider(
                    value: _w2,
                    max: 10,
                    min: -10,
                    onChanged: (value) {
                      setState(() {
                        _w2 = value;
                      });
                    },
                  ),
                ),
              ],
            ),
            Row(
              children: [
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 30),
                  child: const Text("b:"),
                ),
                Flexible(
                  child: Slider(
                    value: _b,
                    max: 200,
                    min: -200,
                    onChanged: (value) {
                      setState(() {
                        _b = value;
                      });
                    },
                  ),
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
      ),
    );
  }
}
