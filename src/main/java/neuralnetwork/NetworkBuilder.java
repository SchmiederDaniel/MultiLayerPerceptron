package neuralnetwork;

import neuralnetwork.initialization.Distribution;
import neuralnetwork.initialization.CustomDistribution;
import neuralnetwork.initialization.NormalDistribution;
import neuralnetwork.initialization.Xavier;
import neuralnetwork.layer.Dense;
import neuralnetwork.layer.Dropout;
import neuralnetwork.layer.Layer;
import neuralnetwork.layer.activation.*;
import neuralnetwork.math.NumpyArray;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
  List<Layer> layerList = new ArrayList<>();
  public Activation activation = new Activation();
  public Layers layer = new Layers();
  public PrivateDistribution distribution = new PrivateDistribution();
  
  NetworkBuilder networkBuilder = this;
  Distribution publicDistribution = new NormalDistribution();
  
  public class PrivateDistribution {
    public NetworkBuilder normalDistribution() {
      publicDistribution = new NormalDistribution();
      return networkBuilder;
    }
    
    public NetworkBuilder zeroBias() {
      publicDistribution = new CustomDistribution(0, 0.0001f, -1, 1);
      return networkBuilder;
    }
    
    public NetworkBuilder ownBias(float from, float to) {
      publicDistribution = new CustomDistribution(from, to, -1, 1);
      return networkBuilder;
    }
    
    public NetworkBuilder customDistribution(float fromBias, float toBias, float fromWeight, float toWeight) {
      publicDistribution = new CustomDistribution(fromBias, toBias, fromWeight, toWeight);
      return networkBuilder;
    }
    
    public NetworkBuilder xavier() {
      publicDistribution = new Xavier(true);
      return networkBuilder;
    }
    
    public NetworkBuilder xavier(boolean normal) {
      publicDistribution = new Xavier(normal);
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
    public NetworkBuilder dense(int input_size, int output_size) {
      layerList.add(new Dense(input_size, output_size, publicDistribution));
      return networkBuilder;
    }
    
    public NetworkBuilder dense(NumpyArray weights, NumpyArray bias) {
      layerList.add(new Dense(weights, bias));
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