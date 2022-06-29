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
  public void trainSingle(double[] realArray, double[] noiseArray, double learning_rate) {
    NumpyArray real = NumpyArray.valueOf(realArray);
    NumpyArray z = NumpyArray.valueOf(noiseArray);
    
    // train Discriminator log(D(real)) + log(1 - D(G(z))
    NumpyArray D_real = predict(discriminatorLayers, real);
    NumpyArray log_D_real = log(D_real);
    
    NumpyArray D_G_z = predict(discriminatorLayers, predict(generatorLayers, z));
    NumpyArray log_1_D_G_z = log(NumpyArray.valueOf(1).subtract(D_G_z));
    
    // multiply by -1 to get gradient ascent?
    NumpyArray grad = log_D_real.add(log_1_D_G_z).multiplyScalar(-1);
    //backward
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
    
    // train Generator log(1 - D(G(z)))
    grad = log_1_D_G_z;
    //backward
    for (int j = generatorLayers.length - 1; j >= 0; j--) {
      Layer layer = generatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
  }
  
  static NumpyArray log(NumpyArray numpyArray) {
    double[][] newData = new double[numpyArray.rows()][numpyArray.cols()];
    for (int rowIndex = 0; rowIndex < numpyArray.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < numpyArray.cols(); colIndex++) {
        newData[rowIndex][colIndex] = Math.log(numpyArray.data[rowIndex][colIndex]);
      }
    }
    return new NumpyArray(newData);
  }
}