package oldneuralnetwork.layer.activation;

import oldneuralnetwork.layer.Layer;
import oldneuralnetwork.NumpyArray;

public class LeakyReLu extends Activation {
  private final float c;
  
  public LeakyReLu(float c, NumpyArray input) {
    super(input);
    this.c = c;
  }
  
  public LeakyReLu() {
    this.c = 0.001f;
  }
  
  public LeakyReLu(float c) {
    this.c = c;
  }
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float x = input.data[rowIndex][colIndex];
        if (x >= 0)
          newData[rowIndex][colIndex] = x;
        else
          newData[rowIndex][colIndex] = x * c;
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
          newData[rowIndex][colIndex] = c;
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    if (this.input == null)
      return new LeakyReLu(c);
    else
      return new LeakyReLu(c, this.input.copy());
  }
  
  @Override
  public String toString() {
    return "LeakyReLu";
  }
}