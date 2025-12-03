package neuralnetwork.layer.activation;

import neuralnetwork.layer.Layer;
import neuralnetwork.math.NumpyArray;

public class Tanh extends Activation {
  public Tanh() {
    
  }
  
  public Tanh(NumpyArray input) {
    super(input);
  }
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][][] newData = new float[input.depth()][input.rows()][input.cols()];
    
    for (int depthIndex = 0; depthIndex < input.depth(); depthIndex++) {
      for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
          float x = input.data[depthIndex][rowIndex][colIndex];
          newData[depthIndex][rowIndex][colIndex] = (float) Math.tanh(x);
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
          x = (float) Math.tanh(x);
          x *= x;
          newData[depthIndex][rowIndex][colIndex] = 1f - x;
        }
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    if (this.input == null)
      return new Tanh();
    else
      return new Tanh(this.input.copy());
  }
}