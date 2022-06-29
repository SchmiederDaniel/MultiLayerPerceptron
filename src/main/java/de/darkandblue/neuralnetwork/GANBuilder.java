package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Dense;
import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.layer.activation.SigmoidActivation;
import de.darkandblue.neuralnetwork.layer.activation.TanhActivation;

import java.util.ArrayList;
import java.util.List;

class GANBuilder {
  List<Layer> generatorLayers = new ArrayList<>();
  List<Layer> discriminatorLayers = new ArrayList<>();
  
  public GANBuilder sigmoidG() {
    generatorLayers.add(new SigmoidActivation());
    return this;
  }
  public GANBuilder sigmoidD() {
    discriminatorLayers.add(new SigmoidActivation());
    return this;
  }
  
  public GANBuilder denseG(int input_size, int output_size) {
    generatorLayers.add(new Dense(input_size, output_size));
    return this;
  }
  public GANBuilder denseD(int input_size, int output_size) {
    discriminatorLayers.add(new Dense(input_size, output_size));
    return this;
  }
  
  public GANBuilder tanhG() {
    generatorLayers.add(new TanhActivation());
    return this;
  }
  public GANBuilder tanhD() {
    discriminatorLayers.add(new TanhActivation());
    return this;
  }
  
  public GANetwork build() {
    return new GANetwork(
      generatorLayers.toArray(new Layer[0]),
      discriminatorLayers.toArray(new Layer[0])
    );
  }
}