package neuralnetwork.initialization;

public class HeNormal extends Distribution {
  int inputSize;
  boolean normal;
  
  public HeNormal(boolean normal) {
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
    // fan_in = inputSize for fully connected or conv layers
    if (normal) {
      double std = Math.sqrt(2f / inputSize);
      return randomizeNormal(0f, (float) std);
    } else {
      // He Uniform limit
      return (float) Math.sqrt(6f / inputSize);
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
    // He initialization does not depend on output size, leave unused
  }
}