import 'package:flutter/material.dart';

import 'data_spread/data_spread.dart';

class DataPoint {
  List<double> inputs;
  late List<double> expectedOutputs;
  late Color pointColor;

  DataPoint(this.inputs, DataSpread dataSpread) {
    dataSpread.setPoint(this);
  }
}