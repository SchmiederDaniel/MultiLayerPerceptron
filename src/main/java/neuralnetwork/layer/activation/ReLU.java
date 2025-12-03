package neuralnetwork.layer.activation;

import neuralnetwork.layer.Layer;
import neuralnetwork.math.NumpyArray;

public class ReLU extends Activation {
  public ReLU(NumpyArray input) {
    super(input);
  }
  
  public ReLU() {
  }
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][][] newData = new float[input.depth()][input.rows()][input.cols()];
    
    for (int depthIndex = 0; depthIndex < input.depth(); depthIndex++) {
      for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
          if (input.data[0][rowIndex][colIndex] > 0) {
            newData[depthIndex][rowIndex][colIndex] = input.data[depthIndex][rowIndex][colIndex];
          } else {
            newData[depthIndex][rowIndex][colIndex] = 0;
          }
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
          newData[depthIndex][rowIndex][colIndex] = input.data[depthIndex][rowIndex][colIndex] > 0 ? 1 : 0;
        }
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public Layer deepCopy() {
    if (this.input == null)
      return new ReLU();
    else
      return new ReLU(this.input.copy());
  }
}