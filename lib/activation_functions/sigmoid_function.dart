import 'dart:math';

import 'package:neural_network/activation_functions/linear_function.dart';

class SigmoidFunction implements LinearFunction {
  @override
  double getActivationOutput(double value) {
    return 1 / (1 + exp(-value));
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return exp(value) / pow((exp(value) + 1), 2);
  }

  @override
  String getName() {
    return "Sigmoid";
  }

}