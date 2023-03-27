package de.darkandblue.neuralnetwork.initialization;

public class NormalDistribution extends Distribution {
  @Override
  public double biasFrom() {
    return -1;
  }
  
  @Override
  public double biasTo() {
    return 1;
  }
  
  @Override
  public double weightsFrom() {
    return -1;
  }
  
  @Override
  public double weightsTo() {
    return 1;
  }
}