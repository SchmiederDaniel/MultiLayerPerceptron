package neuralnetwork.layer.activation;

import neuralnetwork.layer.Layer;
import neuralnetwork.math.NumpyArray;

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
    float[][][] newData = new float[input.depth()][input.rows()][input.cols()];
    
    for (int depthIndex = 0; depthIndex < input.depth(); depthIndex++) {
      for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
          float x = input.data[depthIndex][rowIndex][colIndex];
          if (x >= 0)
            newData[depthIndex][rowIndex][colIndex] = x;
          else
            newData[depthIndex][rowIndex][colIndex] = x * c;
        }
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    float[][][] newData = new float[input.depth()][input.rows()][input.cols()];
    
    for (int depthIndex = 0; depthIndex < input.depth(); depthIndex++) {
      for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
          float x = input.data[depthIndex][rowIndex][colIndex];
          if (x > 0)
            newData[depthIndex][rowIndex][colIndex] = 1;
          else
            newData[depthIndex][rowIndex][colIndex] = c;
        }
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