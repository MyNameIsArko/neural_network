import 'data_spread/data_spread.dart';

class DataPoint {
  List<double> inputs;
  late List<double> expectedOutputs;

  DataPoint(this.inputs, {DataSpread? dataSpread}) {
    if (dataSpread == null) {
      return;
    }
    dataSpread.setPoint(this);
  }
}