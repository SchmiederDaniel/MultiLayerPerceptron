package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Tanh extends Activation {
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float x = input.data[rowIndex][colIndex];
        newData[rowIndex][colIndex] = (float) Math.tanh(x);
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
        x = (float) Math.tanh(x);
        x *= x;
        newData[rowIndex][colIndex] = 1f - x;
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    return new Tanh();
  }
}