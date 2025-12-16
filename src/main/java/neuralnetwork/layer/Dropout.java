package neuralnetwork.layer;

import neuralnetwork.math.Tensor;

public class Dropout extends Layer {
    private final float dropRate;
    private Tensor mask; // Stores which neurons were kept (1) or dropped (0)
    
    /**
     * @param dropRate The probability of dropping a neuron (0.0 to 1.0). 
     * E.g., 0.2 means 20% of neurons are zeroed out.
     */
    public Dropout(float dropRate) {
        if (dropRate < 0 || dropRate >= 1) {
            throw new IllegalArgumentException("Drop rate must be between 0 and 1 (exclusive of 1)");
        }
        this.dropRate = dropRate;
    }

    @Override
    public Tensor forward(Tensor input) {
        // 1. If not training (Inference), pass through exactly as is.
        if (!this.isTraining) {
            return input;
        }

        // 2. Generate Mask (1 = keep, 0 = drop)
        // We abuse applyOperation to generate a tensor of the same shape with random 0s/1s
        // (The input value 'v' is ignored in the lambda, we just use the shape iteration)
        this.mask = input.applyOperation(_ -> Math.random() >= dropRate ? 1.0f : 0.0f);

        // 3. Calculate Scale Factor (Inverted Dropout)
        // If we drop 50%, we double the remaining values to maintain magnitude.
        float scale = 1.0f / (1.0f - dropRate);

        // 4. Output = Input * Mask * Scale
        Tensor masked = input.multiply(this.mask);
        return masked.applyOperation(v -> v * scale);
    }

    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        // Gradient only flows through the neurons that were kept (mask = 1).
        // We must also apply the same scale factor.
        
        float scale = 1.0f / (1.0f - dropRate);
        
        // dL/dx = dL/dy * Mask * Scale
        Tensor grad = outputGradient.multiply(this.mask);
        return grad.applyOperation(v -> v * scale);
    }

    @Override
    public Dropout clone() {
        Dropout cloned = new Dropout(this.dropRate);
        cloned.isTraining = this.isTraining;
        // We do not copy the mask; it is transient state for the current batch
        return cloned;
    }
}