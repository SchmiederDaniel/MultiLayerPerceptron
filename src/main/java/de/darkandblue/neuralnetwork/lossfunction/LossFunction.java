package de.darkandblue.neuralnetwork.lossfunction;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public interface LossFunction {
  public abstract double loss(NumpyArray y_true, NumpyArray y_pred);
  
  public abstract NumpyArray loss_prime(NumpyArray y_true, NumpyArray y_pred);
}