package neuralnetwork.lossfunction;

import neuralnetwork.math.NumpyArray;

public interface LossFunction {
  public abstract float loss(NumpyArray y_true, NumpyArray y_pred);
  
  public abstract NumpyArray loss_prime(NumpyArray y_true, NumpyArray y_pred);
}