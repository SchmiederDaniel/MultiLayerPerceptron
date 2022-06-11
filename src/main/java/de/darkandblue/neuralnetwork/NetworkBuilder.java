package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.activation.SigmoidActivation;
import de.darkandblue.neuralnetwork.layer.activation.TanhActivation;
import de.darkandblue.neuralnetwork.layer.Dense;
import de.darkandblue.neuralnetwork.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
  List<Layer> layerList = new ArrayList<>();
  
  public NetworkBuilder dense(int input_size, int output_size) {
    layerList.add(new Dense(input_size, output_size));
    return this;
  }
  
  public NetworkBuilder sigmoid() {
    layerList.add(new SigmoidActivation());
    return this;
  }
  
  public NetworkBuilder tanh() {
    layerList.add(new TanhActivation());
    return this;
  }
  
  public NeuralNetwork build() {
    
    return new NeuralNetwork(layerList.toArray(new Layer[0]));
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
}