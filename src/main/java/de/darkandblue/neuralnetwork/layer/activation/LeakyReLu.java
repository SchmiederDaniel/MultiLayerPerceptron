package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class LeakyReLu extends Activation {
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float x = input.data[rowIndex][colIndex];
        if (x > 0)
          newData[rowIndex][colIndex] = x;
        else
          newData[rowIndex][colIndex] = x * 0.01f;
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float x = input.data[rowIndex][colIndex];
        if (x > 0)
          newData[rowIndex][colIndex] = 1;
        else
          newData[rowIndex][colIndex] = x / 0.01f;
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    return new LeakyReLu();
  }
}