package oldneuralnetwork.layer.activation;

import oldneuralnetwork.layer.Layer;
import oldneuralnetwork.NumpyArray;

public class ReLU extends Activation {
  public ReLU(NumpyArray input) {
    super(input);
  }
  
  public ReLU() {
  }
  
  @Override
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        if (input.data[rowIndex][colIndex] > 0) {
          newData[rowIndex][colIndex] = input.data[rowIndex][colIndex];
        } else {
          newData[rowIndex][colIndex] = 0;
        }
      }
    }
    
    return new NumpyArray(newData);
  }
  
  @Override
  public NumpyArray activation_prime(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        newData[rowIndex][colIndex] = input.data[rowIndex][colIndex] > 0 ? 1 : 0;
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