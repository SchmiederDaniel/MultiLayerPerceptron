package de.darkandblue.neuralnetwork.lossfunction;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public class OwnLoss implements LossFunction {
  @Override
  public double loss(NumpyArray y_true, NumpyArray y_pred) {
    if (y_true.rows() != y_pred.rows())
      throw new IllegalArgumentException("Rows doesn't match " + y_true.rows() + ", " + y_pred.rows());
    if (y_true.cols() != y_pred.cols())
      throw new IllegalArgumentException("Cols doesn't match " + y_true.cols() + ", " + y_pred.cols());
    
    double sum = 0;
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        sum += Math.abs(y_pred.data[rowIndex][colIndex] - y_true.data[rowIndex][colIndex]);
      }
    }
    return sum / (y_true.rows() * y_pred.cols());

//    return 2 * (y_pred - y_true) / np.size(y_true)
  }
  
  @Override
  public NumpyArray loss_prime(NumpyArray y_true, NumpyArray y_pred) {
    if (y_true.rows() != y_pred.rows())
      throw new IllegalArgumentException("Rows doesn't match " + y_true.rows() + ", " + y_pred.rows());
    if (y_true.cols() != y_pred.cols())
      throw new IllegalArgumentException("Cols doesn't match " + y_true.cols() + ", " + y_pred.cols());
    
    double[][] newData = new double[y_true.rows()][y_pred.cols()];
    
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        newData[rowIndex][colIndex] = y_pred.data[rowIndex][colIndex] - y_pred.data[rowIndex][colIndex];
      }
    }
    return new NumpyArray(newData);
//    return 2 * (y_pred - y_true) / np.size(y_true)
  }
}