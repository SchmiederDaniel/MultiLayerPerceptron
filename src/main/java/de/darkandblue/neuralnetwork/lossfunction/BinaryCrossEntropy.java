package de.darkandblue.neuralnetwork.lossfunction;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public class BinaryCrossEntropy implements LossFunction {
  
  @Override
  public double loss(NumpyArray y_true, NumpyArray y_pred) {
//    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
  
    if (y_true.rows() != y_pred.rows())
      throw new IllegalArgumentException("Rows doesn't match " + y_true.rows() + ", " + y_pred.rows());
    if (y_true.cols() != y_pred.cols())
      throw new IllegalArgumentException("Cols doesn't match " + y_true.cols() + ", " + y_pred.cols());
    
    double sum = 0;
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        sum += -y_true.data[rowIndex][colIndex] * Math.log(y_pred.data[rowIndex][colIndex])
          - (1 - y_true.data[rowIndex][colIndex]) * Math.log(1 - y_pred.data[rowIndex][colIndex]);
      }
    }
    return sum / (y_true.rows() * y_pred.cols());
  }
  
  @Override
  public NumpyArray loss_prime(NumpyArray y_true, NumpyArray y_pred) {
//    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
  
    if (y_true.rows() != y_pred.rows())
      throw new IllegalArgumentException("Rows doesn't match " + y_true.rows() + ", " + y_pred.rows());
    if (y_true.cols() != y_pred.cols())
      throw new IllegalArgumentException("Cols doesn't match " + y_true.cols() + ", " + y_pred.cols());
    
    double[][] newData = new double[y_true.rows()][y_pred.cols()];
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        double pred = y_pred.data[rowIndex][colIndex];
        double target = y_true.data[rowIndex][colIndex];
        newData[rowIndex][colIndex] = ((1d - target) / (1d - pred) - target / pred) / (y_true.rows() * y_true.cols());
      }
    }
    return new NumpyArray(newData);
  }
}