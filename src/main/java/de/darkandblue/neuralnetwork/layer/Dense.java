package de.darkandblue.neuralnetwork.layer;

import de.darkandblue.neuralnetwork.initialization.Distribution;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Dense extends Layer {
  public NumpyArray weights;
  public NumpyArray bias;
  
  public Dense(int input_size, int output_size, Distribution distribution) {
    weights = new NumpyArray(output_size, input_size);
    bias = new NumpyArray(output_size, 1);
    
    weights.randomize(distribution.weightsFrom(), distribution.weightsTo());
    bias.randomize(distribution.biasFrom(), distribution.biasTo());
  }
  
  public Dense(NumpyArray weights, NumpyArray bias, NumpyArray inputCopy) {
    this.weights = weights;
    this.bias = bias;
    
    this.input = inputCopy;
  }
  
  public Dense(NumpyArray weights, NumpyArray bias) {
    this.weights = weights;
    this.bias = bias;
  }
  
  NumpyArray input;
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    return weights.dot(input).add(bias);
  }
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, double learning_rate) {
    // TODO: dimension output_gradient = (1, 1)
    NumpyArray weights_gradient = output_gradient.dot(input.transpose());
    NumpyArray input_gradient = weights.transpose().dot(output_gradient);
    weights = weights.subtract(weights_gradient.multiplyScalar(learning_rate));
    bias = bias.subtract(output_gradient.multiplyScalar(learning_rate)); // idk if activation is a vector or matrix
    return input_gradient;
  }
  
  @Override
  public Layer deepCopy() {
    return new Dense(weights.copy(), bias.copy(), input.copy());
  }
  
  @Override
  public String toString() {
    return "Dense " + weights.dimension() + " " + bias.dimension();
  }
  
//  @Override
//  public String toString() {
//    return "Dense " + weights + " " + bias;
//  }
}