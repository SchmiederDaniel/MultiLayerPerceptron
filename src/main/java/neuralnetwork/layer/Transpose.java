package neuralnetwork.layer;

import neuralnetwork.math.NumpyArray;

/**
 * Transposes the input.
 */
public class Transpose extends Layer {
  @Override
  public NumpyArray forward(NumpyArray input) {
    return input.transpose();
  }
  
  @Override
  public NumpyArray backward(NumpyArray dOut, float lr) {
    return dOut.transpose();
  }
  
  @Override
  public Layer deepCopy() {
    Transpose f = new Transpose();
    return f;
  }
  
  @Override
  public String toString() {
    return "Transpose()";
  }
}