package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.activation.*;
import de.darkandblue.neuralnetwork.layer.Dense;
import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
  List<Layer> layerList = new ArrayList<>();
  
  public NetworkBuilder dense(int input_size, int output_size) {
    layerList.add(new Dense(input_size, output_size));
    return this;
  }
  
  public NetworkBuilder dense(NumpyArray weights, NumpyArray bias) {
    layerList.add(new Dense(weights, bias));
    return this;
  }
  public NeuralNetwork build() {
    
    return new NeuralNetwork(layerList.toArray(Layer[]::new));
  }
  
  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    
    for (Layer layer : layerList) {
      stringBuilder.append(layer.getClass());
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
  
  public NetworkBuilder reLU() {
    layerList.add(new ReLU());
    return this;
  }
  
  public NetworkBuilder sigmoid() {
    layerList.add(new SigmoidActivation());
    return this;
  }
  
  public NetworkBuilder softMax() {
    layerList.add(new SoftMax());
    return this;
  }
  
  public NetworkBuilder tanh() {
    layerList.add(new TanhActivation());
    return this;
  }
  
  public NetworkBuilder elu() {
    layerList.add(new ELU());
    return this;
  }
}