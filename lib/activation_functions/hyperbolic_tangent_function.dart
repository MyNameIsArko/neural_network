import 'dart:math';

import 'package:neural_network/activation_functions/linear_function.dart';

class HyperbolicTangentFunction implements LinearFunction{
  @override
  double getActivationOutput(double value) {
    return (1 - exp(-2 * value)) / (1 + exp(-2 * value));
  }

  @override
  String getName() {
    return "Hyperbolic Tangent";
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return (4 * exp(2 * value)) / pow((exp(2 * value) + 1), 2);
  }
}