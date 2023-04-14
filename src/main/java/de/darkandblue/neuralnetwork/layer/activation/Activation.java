package de.darkandblue.neuralnetwork.layer.activation;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public abstract class Activation extends Layer {
  NumpyArray input;
  
  public Activation(NumpyArray copyInput) {
    this.input = copyInput;
  }
  
  public Activation() {
  }
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    return activation(input);
  }
  
  public abstract NumpyArray activation(NumpyArray input);
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, float learning_rate) {
    return output_gradient.multiply(activation_prime(input));
  }
  
  public abstract NumpyArray activation_prime(NumpyArray input);
}