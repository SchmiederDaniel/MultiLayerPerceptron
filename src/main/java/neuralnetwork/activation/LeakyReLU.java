package neuralnetwork.activation;

import neuralnetwork.math.Tensor;

public class LeakyReLU extends Activation {
  
  private final float alpha; // slope for x < 0
  
  /**
   * Create LeakyReLU with default alpha = 0.01
   */
  public LeakyReLU() {
    this(0.01f);
  }
  
  /**
   * Create LeakyReLU with custom alpha.
   *
   * @param alpha leakage parameter (typical 0.01)
   */
  public LeakyReLU(float alpha) {
    this.alpha = alpha;
  }
  
  private float leaky(float x) {
    return x >= 0f ? x : alpha * x;
  }
  
  @Override
  protected Tensor activation(Tensor input) {
    return input.applyOperation(this::leaky);
  }
  
  @Override
  protected Tensor activationPrime(Tensor input) {
    // derivative:
    // 1 if x >= 0
    // alpha if x < 0
    return input.applyOperation(x -> x >= 0f ? 1f : alpha);
  }
}