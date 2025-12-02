package de.darkandblue.neuralnetwork.layer;

import de.darkandblue.neuralnetwork.math.NumpyArray;

import java.util.Random;

public class Dropout extends Layer {
    private final float dropoutRate;
    private NumpyArray mask;
    private final Random rng = new Random();

    public Dropout(float dropoutRate) {
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new IllegalArgumentException("dropoutRate must be in [0, 1)");
        this.dropoutRate = dropoutRate;
    }

    @Override
    public NumpyArray forward(NumpyArray input) {
        if (isTraining) {
            mask = new NumpyArray(input.rows(), input.cols());
            float keepProb = 1.0f - dropoutRate;

            // Fill mask with 1.0 or 0.0
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.cols(); j++) {
                    // keep neuron with probability (1 - dropoutRate)
                    mask.data[i][j] = rng.nextFloat() < keepProb ? 1f : 0f;
                }
            }

            // Apply dropout + scale to maintain expected value
            NumpyArray dropped = input.multiply(mask);
            dropped = dropped.multiply(1f / keepProb);
            return dropped;
        } else {
            // During inference, do nothing
            return input;
        }
    }

    @Override
    public NumpyArray backward(NumpyArray output_gradient, float learning_rate) {
        if (isTraining && mask != null) {
            float keepProb = 1f - dropoutRate;
            // Backprop only through the kept neurons, scale same as forward
            NumpyArray grad = output_gradient.multiply(mask);
            grad = grad.multiply(1f / keepProb);
            return grad;
        } else {
            return output_gradient;
        }
    }

    @Override
    public Layer deepCopy() {
        Dropout copy = new Dropout(dropoutRate);
        copy.isTraining = this.isTraining;
        if (mask != null)
            copy.mask = mask.copy();
        return copy;
    }

    @Override
    public String toString() {
        return "Dropout(rate=" + dropoutRate + ")";
    }
}