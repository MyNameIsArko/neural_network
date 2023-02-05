import 'package:neural_network/activation_functions/activation_function.dart';

class LinearFunction implements ActivationFunction {
  @override
  double getActivationOutput(double value) {
    return value;
  }

  @override
  String getName() {
    return "Linear";
  }

}