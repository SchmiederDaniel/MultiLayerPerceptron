package de.darkandblue.neuralnetwork.layer;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Dense extends Layer {
  NumpyArray weights;
  NumpyArray bias;
  
  public Dense(int input_size, int output_size) {
    weights = new NumpyArray(output_size, input_size);
    bias = new NumpyArray(output_size, 1);
  
    weights.randomize(-1, 1);
    bias.randomize(-1, 1);
  }
  
  NumpyArray input;
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    return weights.dot(input).add(bias);
  }
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, double learning_rate) {
    System.out.println("teste " + output_gradient.dimension());
    // TODO: dimension output_gradient = (1, 1)
    NumpyArray weights_gradient = output_gradient.dot(input.transpose());
    NumpyArray input_gradient = weights.transpose().dot(output_gradient);
    weights = weights.subtract(weights_gradient.multiplyScalar(learning_rate));
    bias = bias.subtract(output_gradient.multiplyScalar(learning_rate)); // idk if bias is a vector or matrix
    return input_gradient;
  }
}