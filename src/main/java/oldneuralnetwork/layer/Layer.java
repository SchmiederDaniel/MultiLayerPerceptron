package oldneuralnetwork.layer;

import oldneuralnetwork.NumpyArray;

public abstract class Layer {
  public boolean isTraining = true;
  
  public abstract NumpyArray forward(NumpyArray input);
  
  public abstract NumpyArray backward(NumpyArray output_gradient, float learning_rate);
  
  public abstract Layer deepCopy();
}