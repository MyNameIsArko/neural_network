import 'package:neural_network/data_point.dart';
import 'package:neural_network/data_spread/data_spread.dart';

class CornerParabola2D implements DataSpread{
  @override
  void setPoint(DataPoint point) {
    double x = point.inputs[0];
    double y = point.inputs[1];
    if (x < -y - 10 && y < 10 && x < 10) {
      point.expectedOutputs = [1, 0];
    } else {
      point.expectedOutputs = [0, 1];
    }
  }

}