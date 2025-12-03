package neuralnetwork.initialization;

public class Xavier extends Distribution {
  int inputSize, outputSize;
  boolean normal;
  
  public Xavier(boolean normal) {
    super();
    this.normal = normal;
  }
  
  @Override
  public float biasFrom() {
    return 0;
  }
  
  @Override
  public float biasTo() {
    return 0.000001f;
  }
  
  @Override
  public float randomWeight() {
    if (normal) {
      double std = Math.sqrt(2.0 / (inputSize + outputSize));
      return randomizeNormal(0f, (float) std);
    } else {
      return (float) Math.sqrt(6f / (inputSize + outputSize));
    }
  }
  
  @Override
  public float randomBias() {
    return 0;
  }
  
  @Override
  public void setInputSize(int inputSize) {
    this.inputSize = inputSize;
  }
  
  @Override
  public void setOutputSize(int outputSize) {
    this.outputSize = outputSize;
  }
}