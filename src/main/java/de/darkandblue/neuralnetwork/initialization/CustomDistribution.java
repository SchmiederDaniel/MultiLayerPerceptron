package de.darkandblue.neuralnetwork.initialization;

public class CustomDistribution extends Distribution {
  double biasFrom;
  double biasTo;
  double weightFrom;
  double weightTo;
  public CustomDistribution(double biasFrom, double biasTo, double weightFrom, double weightTo) {
    super();
    
    this.biasFrom = biasFrom;
    this.biasTo = biasTo;
    
    this.weightFrom = weightFrom;
    this.weightTo = weightTo;
  }
  
  @Override
  public double biasFrom() {
    return biasFrom;
  }
  
  @Override
  public double biasTo() {
    return biasTo;
  }
  
  @Override
  public double weightsFrom() {
    return weightFrom;
  }
  
  @Override
  public double weightsTo() {
    return weightTo;
  }
}