package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class GANetwork {
  Layer[] generatorLayers;
  Layer[] discriminatorLayers;
  
  public GANetwork(Layer[] generatorLayers, Layer[] discriminatorLayers) {
    this.generatorLayers = generatorLayers;
    this.discriminatorLayers = discriminatorLayers;
  }
  
  private static NumpyArray predict(Layer[] layerArray, NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : layerArray) {
      output = layer.forward(output);
    }
    
    return output;
  }
  
  public NumpyArray predictGeneratorThreadSafe(double... input) {
    return predictGeneratorThreadSafe(NumpyArray.valueOf(input));
  }
  
  public NumpyArray predictGeneratorThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : generatorLayers) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  // Own train function
  public void trainSingle(LossFunction lossFunction, double[] realArray, double[] noiseArray, double learning_rate) {
    NumpyArray x = NumpyArray.valueOf(realArray);
    trainDiscriminator(lossFunction, x, NumpyArray.valueOf(0), learning_rate);
  
    NumpyArray z = NumpyArray.valueOf(noiseArray);
    
    //forward
    NumpyArray generated = predict(generatorLayers, z);
    NumpyArray errorDiscriminator = trainDiscriminator(lossFunction, generated, NumpyArray.valueOf(1), learning_rate);
  
    //backward
    NumpyArray grad = lossFunction.loss_prime(NumpyArray.valueOf(0), errorDiscriminator);
    for (int i = discriminatorLayers.length - 1; i >= 0; i--) {
      Layer layer = discriminatorLayers[i];
      grad = layer.backward(grad, learning_rate);
    }
  
    grad = lossFunction.loss_prime(grad, generated).multiplyScalar(-1);
    for (int i = generatorLayers.length - 1; i >= 0; i--) {
      Layer layer = generatorLayers[i];
      grad = layer.backward(grad, learning_rate);
    }
  }
  
  private NumpyArray trainDiscriminator(LossFunction lossFunction, NumpyArray input, NumpyArray target, double learning_rate) {
    //forward
    NumpyArray output = predict(discriminatorLayers, input);
  
    //backward
    NumpyArray grad = lossFunction.loss_prime(target, output);
  
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
    
    return output;
  }
}