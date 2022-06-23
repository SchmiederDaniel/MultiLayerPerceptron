package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class SoftMax extends Layer {
  NumpyArray input;
  NumpyArray output;
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    return activation(input);
  }
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, double learning_rate) {
    return output_gradient.multiplyScalar(activation_prime(output_gradient));
  }
  
  public NumpyArray activation(NumpyArray input) {
    double[][] newData = new double[input.rows()][input.cols()];
    double sum = 0;
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        sum += Math.exp(input.data[rowIndex][colIndex]);
      }
    }
    
    for (int rowIndex = 0; rowIndex < input.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < input.cols(); colIndex++) {
        double x = input.data[rowIndex][colIndex];
        newData[rowIndex][colIndex] = Math.exp(x) / sum;
      }
    }
    
    output = new NumpyArray(newData);
    return new NumpyArray(newData);
  }
  
  public NumpyArray activation_prime(NumpyArray output_gradient) {
    NumpyArray identity = NumpyArray.identity(output.rows() * output.cols());
    return identity.subtract(output.transpose()).multiplyScalar(output).dot(output_gradient);
  }
}