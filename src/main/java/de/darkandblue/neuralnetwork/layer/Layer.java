package de.darkandblue.neuralnetwork.layer;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public abstract class Layer {
  public abstract NumpyArray forward(NumpyArray input);
  
  public abstract NumpyArray backward(NumpyArray output_gradient, double learning_rate);
  
  public abstract Layer copy();
}