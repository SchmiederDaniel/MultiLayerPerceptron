package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class ELU extends Activation {
  private final static double alpha = 1;
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        double x = input.data[rowIndex][colIndex];
        if (x >= 0)
          newData[rowIndex][colIndex] = x;
        else
          newData[rowIndex][colIndex] = alpha * (Math.pow(Math.E, x) - 1);
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
        if (x > 0)
          newData[rowIndex][colIndex] = 1;
        else
          newData[rowIndex][colIndex] = alpha * Math.exp(x);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  public static double sechPow2(double x) {
    return 4 * coshPow2(x) / Math.pow(Math.cosh(2 * x) + 1, 2);
  }
  
  public static double coshPow2(double x) {
    return 0.5 * (Math.cosh(2 * x) + 1);
  }
  
  @Override
  public Layer deepCopy() {
    return new ELU();
  }
}