package neuralnetwork.lossfunction;

import neuralnetwork.math.NumpyArray;

public class MeanSquareError implements LossFunction {
  // Mean square error loss function
  @Override
  public float loss(NumpyArray y_true, NumpyArray y_pred) {
    if (y_true.rows() != y_pred.rows())
      throw new IllegalArgumentException("Rows doesn't match " + y_true.rows() + ", " + y_pred.rows());
    if (y_true.cols() != y_pred.cols())
      throw new IllegalArgumentException("Cols doesn't match " + y_true.cols() + ", " + y_pred.cols());
    
    float sum = 0;
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        sum += Math.pow(y_true.data[rowIndex][colIndex] - y_pred.data[rowIndex][colIndex], 2);
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
  
    float[][] newData = new float[y_true.rows()][y_pred.cols()];
    
    for (int rowIndex = 0; rowIndex < y_true.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < y_true.cols(); colIndex++) {
        newData[rowIndex][colIndex] = 2f * (y_pred.data[rowIndex][colIndex] - y_true.data[rowIndex][colIndex]) / (y_true.rows() * y_true.cols());
      }
    }
    return new NumpyArray(newData);
//    return 2 * (y_pred - y_true) / np.size(y_true)
  }
}