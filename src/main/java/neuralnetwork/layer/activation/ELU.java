package neuralnetwork.layer.activation;

import neuralnetwork.layer.Layer;
import neuralnetwork.math.NumpyArray;

public class ELU extends Activation {
  private final static float alpha = 1;
  
  public ELU(NumpyArray input) {
    super(input);
  }
  
  public ELU() {
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
            newData[depthIndex][rowIndex][colIndex] = (float) (alpha * (Math.pow(Math.E, x) - 1));
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
            newData[depthIndex][rowIndex][colIndex] = (float) (alpha * Math.exp(x));
        }
      }
    }
    
    return new NumpyArray(newData);
  }
  
  public static float sechPow2(float x) {
    return (float) (4 * coshPow2(x) / Math.pow(Math.cosh(2 * x) + 1, 2));
  }
  
  public static float coshPow2(float x) {
    return (float) (0.5 * (Math.cosh(2 * x) + 1));
  }
  
  @Override
  public Layer deepCopy() {
    if (this.input == null)
      return new ELU();
    else
      return new ELU(this.input.copy());
  }
}