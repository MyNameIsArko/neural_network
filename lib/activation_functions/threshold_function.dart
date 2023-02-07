import 'package:neural_network/activation_functions/linear_function.dart';

class ThresholdFunction implements LinearFunction {
  @override
  double getActivationOutput(double value) {
    return value >= 0 ? 1 : 0;
  }

  @override
  double getActivationDerivativeOutput(double value) {
    return 0;
  }

  @override
  String getName() {
    return "Threshold";
  }

}