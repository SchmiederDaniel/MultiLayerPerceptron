package neuralnetwork;

import neuralnetwork.initialization.Distribution;
import neuralnetwork.initialization.CustomDistribution;
import neuralnetwork.initialization.NormalDistribution;
import neuralnetwork.initialization.Xavier;
import neuralnetwork.layer.*;
import neuralnetwork.layer.activation.*;
import neuralnetwork.math.NumpyArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class NetworkBuilder {
  List<Layer> layerList = new ArrayList<>();
  public Activation activation = new Activation();
  public Layers layer = new Layers();
  public PrivateDistribution distribution = new PrivateDistribution();
  
  NetworkBuilder networkBuilder = this;
  Distribution weightInitialization = new Xavier(true);
  private int[] lastConvDimensions;
  
  public class PrivateDistribution {
    public NetworkBuilder normalDistribution() {
      weightInitialization = new NormalDistribution();
      return networkBuilder;
    }
    
    public NetworkBuilder zeroBias() {
      weightInitialization = new CustomDistribution(0, 0.0001f, -1, 1);
      return networkBuilder;
    }
    
    public NetworkBuilder ownBias(float from, float to) {
      weightInitialization = new CustomDistribution(from, to, -1, 1);
      return networkBuilder;
    }
    
    public NetworkBuilder customDistribution(float fromBias, float toBias, float fromWeight, float toWeight) {
      weightInitialization = new CustomDistribution(fromBias, toBias, fromWeight, toWeight);
      return networkBuilder;
    }
    
    public NetworkBuilder xavier() {
      weightInitialization = new Xavier(true);
      return networkBuilder;
    }
    
    public NetworkBuilder xavier(boolean normal) {
      weightInitialization = new Xavier(normal);
      return networkBuilder;
    }
  }
  
  public class Activation {
    public NetworkBuilder reLU() {
      layerList.add(new ReLU());
      return networkBuilder;
    }
    
    public NetworkBuilder sigmoid() {
      layerList.add(new Sigmoid());
      return networkBuilder;
    }
    
    public NetworkBuilder softMax() {
      layerList.add(new SoftMax());
      return networkBuilder;
    }
    
    public NetworkBuilder tanh() {
      layerList.add(new Tanh());
      return networkBuilder;
    }
    
    public NetworkBuilder elu() {
      layerList.add(new ELU());
      return networkBuilder;
    }
    
    public NetworkBuilder leakyReLu() {
      layerList.add(new LeakyReLu());
      return networkBuilder;
    }
  }
  
  public class Layers {
    
    private int multiply(int[] values) {
      return Arrays.stream(values).reduce(1, (a, b) -> a * b);
    }
    
    public NetworkBuilder dense(int output_size) {
      if (lastConvDimensions == null) {
        throw new IllegalStateException("No convolutional layer found before dense layer.");
      }
      int input_size = multiply(lastConvDimensions);
      System.out.println("creation: " + input_size);
      lastConvDimensions = null;
      layerList.add(new Dense(input_size, output_size, weightInitialization));
      return networkBuilder;
    }
    
    public NetworkBuilder dense(int input_size, int output_size) {
      layerList.add(new Dense(input_size, output_size, weightInitialization));
      return networkBuilder;
    }
    
    public NetworkBuilder dense(NumpyArray weights, NumpyArray bias) {
      layerList.add(new Dense(weights, bias));
      return networkBuilder;
    }
    
    public NetworkBuilder conv2D(int inputHeight, int inputWidth, int kernelHeight, int kernelWidth, int strideH, int strideW, int filters) {
      Conv2D layer = new Conv2D(filters, kernelHeight, kernelWidth, strideH, strideW, weightInitialization);
      lastConvDimensions = layer.outputDimension(inputHeight, inputWidth);
      layerList.add(layer);
      return networkBuilder;
    }
    
    public NetworkBuilder conv2D(int inputHeight, int inputWidth, int kernelHeight, int kernelWidth, int stride, int filters) {
      return conv2D(inputHeight, inputWidth, kernelHeight, kernelWidth, stride, stride, filters);
    }
    
    public NetworkBuilder flatten() {
      layerList.add(new Flatten());
      return networkBuilder;
    }
    
    public NetworkBuilder transpose() {
      layerList.add(new Transpose());
      return networkBuilder;
    }
    
    public NetworkBuilder dropOut(float dropOutRate) {
      layerList.add(new Dropout(dropOutRate));
      return networkBuilder;
    }
    
    public NetworkBuilder dropOut(double dropOutRate) {
      return dropOut((float) dropOutRate);
    }
  }
  
  public NeuralNetwork build() {
    return new NeuralNetwork(layerList.toArray(new Layer[0]));
  }
  
  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    
    for (Layer layer : layerList) {
      stringBuilder.append(layer);
      stringBuilder.append(", ");
    }
    
    String outputString;
    if (layerList.size() == 0) {
      stringBuilder.append("Empty");
      outputString = stringBuilder.toString();
    } else {
      outputString = stringBuilder.substring(0, stringBuilder.length() - 2);
    }
    
    return outputString;
  }
}