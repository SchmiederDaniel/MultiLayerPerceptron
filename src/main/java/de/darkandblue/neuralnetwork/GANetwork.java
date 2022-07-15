package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.BasicLoss;
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
    return predictGeneratorThreadSafe(NumpyArray.of(input));
  }
  
  public NumpyArray predictGeneratorThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : generatorLayers) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  public NumpyArray predictDiscriminatorThreadSafe(double... input) {
    return predictDiscriminatorThreadSafe(NumpyArray.of(input));
  }
  
  public NumpyArray predictDiscriminatorThreadSafe(NumpyArray input) {
    NumpyArray output = input;
    for (Layer layer : discriminatorLayers) {
      layer = layer.deepCopy();
      output = layer.forward(output);
    }
    return output;
  }
  
  public double getGeneratorLoss(double[] noise, double[] targets) {
    NumpyArray y = NumpyArray.of(targets);
    NumpyArray x = predictGeneratorThreadSafe(noise);
    
//    return .data;
    return average(y.subtract(x).transpose().data);
  }
  
  double average(double[][] array) {
    int count = 0;
    double sum = 0;
    for (double[] doubles : array) {
      for (double aDouble : doubles) {
        count++;
        sum += Math.abs(aDouble);
      }
    }
    return sum / count;
  }
  
  public double getDiscriminatorLoss(double[] real, double[] targets) {
    NumpyArray y = NumpyArray.of(targets);
    NumpyArray a = predictDiscriminatorThreadSafe(real);
    
//    return y.subtract(a).data;
    return average(y.subtract(a).transpose().data);
  }
  
  public void trainOwnSingle(double[] realArray, double[] fakeArray, double[] noiseArray, double learning_rate) {
    double[] targets = new double[] { 0 };
    NumpyArray x = NumpyArray.of(realArray);
    NumpyArray y = predict(discriminatorLayers, x);
    NumpyArray grad = new BasicLoss().loss_prime(NumpyArray.of(targets), y);
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
  
    targets = new double[] { 1 };
    x = NumpyArray.of(fakeArray);
    y = predict(discriminatorLayers, x);
    grad = new BasicLoss().loss_prime(NumpyArray.of(targets), y);
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
  
    NumpyArray generatorPredict = predict(generatorLayers, NumpyArray.of(noiseArray));
    NumpyArray discriminatorPredict = predict(discriminatorLayers, generatorPredict);
    
    targets = new double[] { 1 };
    grad = new BasicLoss().loss_prime(NumpyArray.of(targets), discriminatorPredict);
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
  
//    grad = NumpyArray.of(1).subtract(grad);
    grad = grad.multiplyScalar(-1);
    for (int j = generatorLayers.length - 1; j >= 0; j--) {
      Layer layer = generatorLayers[j];
      grad = layer.backward(grad, learning_rate * 3);
    }
  }
  
  // Own train function
  public void trainSingle(double[] realArray, double[] noiseArray, double learning_rate) {
    NumpyArray x = NumpyArray.of(realArray);
    NumpyArray z = NumpyArray.of(noiseArray);
    
    // train Discriminator log(D(x)) + log(1 - D(G(z))
    NumpyArray D_x = predict(discriminatorLayers, x);
    NumpyArray log_D_x = log(D_x);
    
    NumpyArray D_G_z = predict(discriminatorLayers, predict(generatorLayers, z));
    NumpyArray log_1_D_G_z = log(NumpyArray.of(1).subtract(D_G_z));
    
    // multiply by -1 to get gradient ascent?
    NumpyArray grad = NumpyArray.of(1).subtract(log_D_x.add(log_1_D_G_z)).multiplyScalar(-1);
    //backward
    for (int j = discriminatorLayers.length - 1; j >= 0; j--) {
      Layer layer = discriminatorLayers[j];
      grad = layer.backward(grad, learning_rate);
    }
    
    // train Generator log(1 - D(G(z)))
//    grad = NumpyArray.valueOf(1).subtract(log_1_D_G_z).multiplyScalar(-1);
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