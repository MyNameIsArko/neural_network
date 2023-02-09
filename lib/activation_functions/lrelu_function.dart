import 'dart:math';

import 'package:neural_network/activation_functions/linear_function.dart';

class LReLUFunction implements LinearFunction {
  @override
  double getActivationOutput(double value) {
    return max(0.1 * value, value);
  }

  @override
  String getName() {
    return "Leaky ReLU";
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return value > 0 ? 1 : 0.1;
  }

}