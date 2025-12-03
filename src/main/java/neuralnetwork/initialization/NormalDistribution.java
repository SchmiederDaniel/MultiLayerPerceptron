package neuralnetwork.initialization;

import java.util.concurrent.ThreadLocalRandom;

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
  public float randomWeight() {
    return ThreadLocalRandom.current().nextFloat(-1, 1);
  }
  
  @Override
  public float randomBias() {
    return ThreadLocalRandom.current().nextFloat(-1, 1);
  }
  
  @Override
  public void setInputSize(int inputSize) {
    
  }
  
  @Override
  public void setOutputSize(int outputSize) {
    
  }
}