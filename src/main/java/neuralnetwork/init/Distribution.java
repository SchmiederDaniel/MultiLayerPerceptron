package neuralnetwork.init;

import java.util.Random;

public abstract class Distribution {
  public abstract float biasFrom();
  
  public abstract float biasTo();
  
  public abstract float randomWeight();
  
  public abstract float randomBias();
  
  public abstract void setInputSize(int inputSize);
  
  public abstract void setOutputSize(int outputSize);
  
  public float randomizeNormal(float mean, float std) {
    Random rng = new Random();
    
    // Box–Muller transform to get a normal distribution
    float u1 = 1f - rng.nextFloat();
    float u2 = 1f - rng.nextFloat();
    float z = (float) (Math.sqrt(-2d * Math.log(u1)) * Math.cos(2d * Math.PI * u2));
    
    return mean + std * z;
  }
}