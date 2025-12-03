package neuralnetwork.initialization;

import java.util.concurrent.ThreadLocalRandom;

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
  public float randomWeight() {
    return ThreadLocalRandom.current().nextFloat(weightFrom, weightTo);
  }
  
  @Override
  public float randomBias() {
    return ThreadLocalRandom.current().nextFloat(biasFrom, biasTo);
  }
  
  @Override
  public void setInputSize(int inputSize) {
    
  }
  
  @Override
  public void setOutputSize(int outputSize) {
    
  }
}