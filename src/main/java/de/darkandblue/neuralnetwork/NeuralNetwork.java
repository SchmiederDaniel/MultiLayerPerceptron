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
    return predict(NumpyArray.numpyArrayOf1DimArray(input));
  }
  
  private NumpyArray predict(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      output = layer.forward(output);
    }
    return output;
  }
  
  public NumpyArray predictThreadSafe(double... input) {
    return predictThreadSafe(NumpyArray.numpyArrayOf1DimArray(input));
  }
  public NumpyArray predictThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  public void train(LossFunction lossFunction, double[][] x_train, double[][] y_train, int epochs, double learning_rate, boolean verbose) {
    if (x_train.length != y_train.length)
      throw new IllegalArgumentException("Length of input and target doesn't match " + x_train.length + " " + y_train.length);
    
    for (int e = 0; e < epochs; e++) {
      double error = 0;
//      for x, y in zip(x_train, y_train):
      for (int i = 0; i < x_train.length; i++) {
        NumpyArray x = NumpyArray.numpyArrayOf1DimArray(x_train[i]);
        NumpyArray y = NumpyArray.numpyArrayOf1DimArray(y_train[i]);
        
        //forward
        NumpyArray output = predict(x);
        
        //error
        error += lossFunction.loss(y, output);
        
        //backward
        NumpyArray grad = lossFunction.loss_prime(y, output);
        
        for (int j = layerArray.length - 1; j >= 0; j--) {
          Layer layer = layerArray[j];
          grad = layer.backward(grad, learning_rate);
        }
      }
      
      error /= x_train.length;
      if (verbose)
        System.out.println("{" + (e + 1d) / epochs + "} error=" + error);
    }
  }
  
  // Own train function
  public void trainSingle(LossFunction lossFunction, double[] x_train, double[] y_train, double learning_rate, boolean verbose) {
    double error = 0;
    NumpyArray x = NumpyArray.numpyArrayOf1DimArray(x_train);
    NumpyArray y = NumpyArray.numpyArrayOf1DimArray(y_train);
    
    //forward
    NumpyArray output = predict(x);
    
    //error
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