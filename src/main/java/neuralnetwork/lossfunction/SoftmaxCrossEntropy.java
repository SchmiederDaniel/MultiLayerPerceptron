package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

public class SoftmaxCrossEntropy implements LossFunction {
    
    private static final float EPS = 1e-9f;
    
    /**
     * Stable softmax for a 1D vector.
     */
    private Tensor softmax(Tensor logits) {
        if (logits.rank() != 1)
            throw new IllegalArgumentException("SoftmaxCrossEntropy expects 1D vectors, got rank=" + logits.rank());
        
        float[] in = logits.toFlatArray();
        int n = in.length;
        float[] out = new float[n];
        
        // 1. Max element for numerical stability
        float max = Float.NEGATIVE_INFINITY;
        for (float v : in) max = Math.max(max, v);
        
        // 2. Exp and sum
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.exp(in[i] - max);
            sum += out[i];
        }
        
        // 3. Normalize
        sum += EPS;
        for (int i = 0; i < n; i++) out[i] /= sum;
        
        return Tensor.createFromFlat(out, n);
    }
    
    @Override
    public float loss(Tensor yTrue, Tensor yPred) {
        
        Tensor probs = softmax(yPred);
        
        // Clip for log stability
        Tensor clipped = probs.applyOperation(v ->
            Math.max(EPS, Math.min(1f - EPS, v))
        );
        
        Tensor log = clipped.applyOperation(v -> (float) Math.log(v));   // log(p)
        Tensor term = yTrue.multiply(log);                               // yTrue * log(p)
        
        float total = term.sum();   // negative
        return -total;              // mean = itself (no batch)
    }
    
    @Override
    public Tensor lossPrime(Tensor yTrue, Tensor yPred) {
        
        Tensor probs = softmax(yPred);
        
        // Gradient: softmax - yTrue
        return probs.subtract(yTrue);
    }
}