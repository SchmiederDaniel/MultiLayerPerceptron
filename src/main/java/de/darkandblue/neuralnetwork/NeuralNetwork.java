package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;

public class NeuralNetwork {
  //  List<Layer> layerList;
  public Layer[] layerArray;
  
  public NeuralNetwork(Layer[] layerArray) {
    this.layerArray = layerArray;
  }
  
  /*
  Should only be used for training and can cause problems when used asynchron
   */
  @Deprecated
  public NumpyArray predict(float... input) {
    return predict(NumpyArray.of(input));
  }
  
  public NumpyArray predict(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      output = layer.forward(output);
    }
    return output;
  }
  
  public NumpyArray predictThreadSafe(float... input) {
    return predictThreadSafe(NumpyArray.of(input));
  }
  
  public NumpyArray predictThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  public NumpyArray backwardWithoutTrain(LossFunction lossFunction, NumpyArray output, NumpyArray y, float learning_rate) {
    NumpyArray grad = lossFunction.loss_prime(y, output);
    for (int j = layerArray.length - 1; j >= 0; j--) {
      Layer layer = layerArray[j].deepCopy();
      grad = layer.backward(grad, learning_rate);
    }
    return grad;
  }
  
  public void trainSingle(LossFunction lossFunction, float[] x_train, float[] y_train, float learning_rate) {
    trainSingle(lossFunction, x_train, y_train, learning_rate, false);
  }
  
  // Own train function
  public void trainSingle(LossFunction lossFunction, float[] x_train, float[] y_train, float learning_rate, boolean verbose) {
    float error = 0;
    NumpyArray x = NumpyArray.of(x_train);
    NumpyArray y = NumpyArray.of(y_train);
    
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
  
  public NumpyArray trainWithoutPredict(LossFunction lossFunction, NumpyArray output, NumpyArray y, float learning_rate) {
    NumpyArray grad = lossFunction.loss_prime(y, output);
    for (int j = layerArray.length - 1; j >= 0; j--) {
      Layer layer = layerArray[j];
      grad = layer.backward(grad, learning_rate);
    }
    return grad;
  }
  
  public NumpyArray trainWithoutPredict(LossFunction lossFunction, NumpyArray output, float[] y_train, float learning_rate) {
    NumpyArray y = NumpyArray.of(y_train);
    return trainWithoutPredict(lossFunction, output, y, learning_rate);
  }
  
  public NeuralNetwork copyMerge(NeuralNetwork neuralNetwork) {
    throw new RuntimeException("not implemented yet L");
  }
  
  public NeuralNetwork copyMutate(float from, float to) {
    throw new RuntimeException("not implemented yet L");
  }
  
  public NeuralNetwork copy() {
    throw new RuntimeException("not implemented yet L");
  }
}