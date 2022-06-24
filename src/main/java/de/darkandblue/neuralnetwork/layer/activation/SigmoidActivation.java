package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class SigmoidActivation extends Activation {
  @Override
  public NumpyArray activation(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        newData[rowIndex][colIndex] = sigmoid(input.data[rowIndex][colIndex]);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  private static double sigmoid(double x) {
    return 1d / (1d + Math.exp(-x));
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        double s = sigmoid(input.data[rowIndex][colIndex]);
        newData[rowIndex][colIndex] = s * (1 - s);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    return new SigmoidActivation();
  }
}