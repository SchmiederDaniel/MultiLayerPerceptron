package de.darkandblue.neuralnetwork.initialization;

public class NormalDistribution extends Distribution {
  @Override
  public float biasFrom() {
    return -1;
  }
  
  @Override
  public float biasTo() {
    return 1;
  }
  
  @Override
  public float weightsFrom() {
    return -1;
  }
  
  @Override
  public float weightsTo() {
    return 1;
  }
}