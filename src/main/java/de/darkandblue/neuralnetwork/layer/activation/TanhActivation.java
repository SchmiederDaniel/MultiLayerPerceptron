package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class TanhActivation extends Activation {
  @Override
  public NumpyArray activation(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        double x = input.data[rowIndex][colIndex];
        newData[rowIndex][colIndex] = Math.tanh(x);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        double x = input.data[rowIndex][colIndex];
        x = Math.tanh(x);
        x *= x;
        newData[rowIndex][colIndex] = 1d - x;
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    return new TanhActivation();
  }
}