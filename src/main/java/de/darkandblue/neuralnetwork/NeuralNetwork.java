package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class NeuralNetwork {
  //  List<Layer> layerList;
  Layer[] layerArray;
  
  public NeuralNetwork(Layer[] layerArray) {
    this.layerArray = layerArray;
  }
  
  public NumpyArray predict(double... input) {
    return predict(NumpyArray.valueOf(input));
  }
  
  private NumpyArray predict(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      output = layer.forward(output);
    }
    
    return output;
  }
  
  public NumpyArray predictThreadSafe(double... input) {
    return predictThreadSafe(NumpyArray.valueOf(input));
  }
  
  public NumpyArray predictThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  // Own train function
  public void trainSingle(LossFunction lossFunction, double[] x_train, double[] y_train, double learning_rate, boolean verbose) {
    double error = 0;
    NumpyArray x = NumpyArray.valueOf(x_train);
    NumpyArray y = NumpyArray.valueOf(y_train);
    
    //forward
    NumpyArray output = predict(x);
    
    //error
    if (verbose)
      error += lossFunction.loss(y, output);
    
    //backward
    NumpyArray grad = lossFunction.loss_prime(y, output);
    
    for (int j = layerArray.length - 1; j >= 0; j--) {
      Layer layer = layerArray[j];
      grad = layer.backward(grad, learning_rate);
    }
    
    if (verbose)
      System.out.println("error=" + error);
  }
}