package neuralnetwork.init;

import java.util.Random;

public class SeededDistribution extends Distribution {
    private final int seed;
    
    public SeededDistribution(int seed) {
        this.seed = seed;
    }
    
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
        Random random = new Random(seed);
        return random.nextFloat(-1, 1);
    }
    
    @Override
    public float randomBias() {
        Random random = new Random(seed);
        return random.nextFloat(-1, 1);
    }
    
    @Override
    public void setInputSize(int inputSize) {
        
    }
    
    @Override
    public void setOutputSize(int outputSize) {
        
    }
}