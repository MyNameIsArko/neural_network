import 'dart:math';

import 'package:neural_network/activation_functions/linear_function.dart';

class ELUFunction implements LinearFunction {
  @override
  double getActivationOutput(double value) {
    return value > 0 ? value : exp(value) - 1;
  }

  @override
  String getName() {
    return "ELU";
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return value > 0 ? 1 : exp(value);
  }

}