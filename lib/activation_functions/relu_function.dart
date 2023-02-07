import 'dart:math';

import 'package:neural_network/activation_functions/linear_function.dart';

class ReLUFunction implements LinearFunction {
  @override
  double getActivationOutput(double value) {
    return max(0, value);
  }

  @override
  String getName() {
    return "ReLU";
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return value > 0 ? 1 : 0;
  }

}