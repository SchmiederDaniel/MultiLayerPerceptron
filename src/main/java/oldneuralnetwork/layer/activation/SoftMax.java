package oldneuralnetwork.layer.activation;

import oldneuralnetwork.layer.Layer;
import oldneuralnetwork.NumpyArray;

public class SoftMax extends Layer {
  NumpyArray input;
  NumpyArray output;
  
  public SoftMax() {
    
  }
  
  public SoftMax(NumpyArray input, NumpyArray output) {
    super();
    this.input = input;
    this.output = output;
  }
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    return activation(input);
  }
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, float learning_rate) {
    return output_gradient.multiply(activation_prime(output_gradient));
  }
  
  @Override
  public Layer deepCopy() {
    return new SoftMax(this.input, this.output);
  }
  
  public NumpyArray activation(NumpyArray input) {
    float[][] newData = new float[input.rows()][input.cols()];
    float sum = 0;
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        sum += Math.exp(input.data[rowIndex][colIndex]);
      }
    }
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        float x = input.data[rowIndex][colIndex];
        newData[rowIndex][colIndex] = (float) (Math.exp(x) / sum);
      }
    }
    
    output = new NumpyArray(newData);
    return new NumpyArray(newData);
  }
  
  public NumpyArray activation_prime(NumpyArray output_gradient) {
    NumpyArray identity = NumpyArray.identity(output.rows() * output.cols());
    return identity.subtract(output.transpose()).multiply(output).dot(output_gradient);
  }
}