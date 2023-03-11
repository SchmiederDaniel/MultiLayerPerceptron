package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.initialization.Distribution;
import de.darkandblue.neuralnetwork.initialization.CustomDistribution;
import de.darkandblue.neuralnetwork.initialization.NormalDistribution;
import de.darkandblue.neuralnetwork.layer.Dense;
import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.layer.activation.*;
import de.darkandblue.neuralnetwork.math.NumpyArray;

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
      publicDistribution = new CustomDistribution(0, 0.0001d, -1, 1);
      return networkBuilder;
    }
  
    public NetworkBuilder ownBias(double from, double to) {
      publicDistribution = new CustomDistribution(from, to, -1, 1);
      return networkBuilder;
    }
  
    public NetworkBuilder customDistribution(double fromBias, double toBias, double fromWeight, double toWeight) {
      publicDistribution = new CustomDistribution(fromBias, toBias, fromWeight, toWeight);
      return networkBuilder;
    }
  }
  
  public class Activation {
    public NetworkBuilder reLU() {
      layerList.add(new ReLU());
      return networkBuilder;
    }
    
    public NetworkBuilder sigmoid() {
      layerList.add(new SigmoidActivation());
      return networkBuilder;
    }
    
    public NetworkBuilder softMax() {
      layerList.add(new SoftMax());
      return networkBuilder;
    }
    
    public NetworkBuilder tanh() {
      layerList.add(new TanhActivation());
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
  }
  
  public de.darkandblue.neuralnetwork.NeuralNetwork build() {
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