package neuralnetwork.layer;

import neuralnetwork.math.Tensor;

/**
 * Base class for all layers.
 *
 * Now implements {@link Cloneable} so networks can be safely cloned for
 * multi-threaded inference or training without sharing mutable state.
 */
public abstract class Layer implements Cloneable {
    public boolean isTraining = true;

    public abstract Tensor forward(Tensor input);

    public abstract Tensor backward(Tensor outputGradient, float learningRate);

    /**
     * Create a clone of this layer. Default implementation performs a shallow clone
     * using {@code Object.clone()}. Subclasses should override to deep copy any
     * internal mutable state (e.g., weights, buffers) and may call {@code super.clone()}
     * to start from a shallow copy.
     */
    @Override
    public Layer clone() {
        try {
            return (Layer) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }
}