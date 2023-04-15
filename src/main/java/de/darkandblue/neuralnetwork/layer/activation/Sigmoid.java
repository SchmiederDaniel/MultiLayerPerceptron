package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Sigmoid extends Activation {
  public Sigmoid(NumpyArray copyInput) {
    super(copyInput);
  }
  
  public Sigmoid() {
  }
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        newData[rowIndex][colIndex] = sigmoid(input.data[rowIndex][colIndex]);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  private static float sigmoid(float x) {
    return (float) (1f / (1f + Math.exp(-x)));
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float s = sigmoid(input.data[rowIndex][colIndex]);
        newData[rowIndex][colIndex] = s * (1 - s);
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    return new Sigmoid(this.input.copy());
  }
  
  @Override
  public String toString() {
    return "Sigmoid";
  }
}