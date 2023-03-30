package de.darkandblue.neuralnetwork.initialization;

public class CustomDistribution extends Distribution {
  float biasFrom;
  float biasTo;
  float weightFrom;
  float weightTo;
  public CustomDistribution(float biasFrom, float biasTo, float weightFrom, float weightTo) {
    super();
    
    this.biasFrom = biasFrom;
    this.biasTo = biasTo;
    
    this.weightFrom = weightFrom;
    this.weightTo = weightTo;
  }
  
  @Override
  public float biasFrom() {
    return biasFrom;
  }
  
  @Override
  public float biasTo() {
    return biasTo;
  }
  
  @Override
  public float weightsFrom() {
    return weightFrom;
  }
  
  @Override
  public float weightsTo() {
    return weightTo;
  }
}