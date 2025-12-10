package neuralnetwork;

import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.math.Tensor;
import neuralnetwork.layer.Layer;

/**
 * A simple feed-forward neural network that uses the new math.tensor API (Vector/Matrix/Tensor).
 */
public class NeuralNetwork implements Cloneable {
    public Layer[] layers;
    
    public NeuralNetwork(Layer... layers) {
        this.layers = layers;
    }
    
    public Tensor predict(Tensor input) {
        Tensor out = input;
        for (Layer layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }
    
    /**
     * Thread-safe prediction: uses a deep-cloned network instance so concurrent
     * training cannot mutate the parameters used for this inference call.
     */
    public Tensor predictThreadSafe(Tensor input) {
        NeuralNetwork copy = this.clone();
        // Ensure inference mode
        for (Layer l : copy.layers) l.isTraining = false;
        return copy.predict(input);
    }
    
    /**
     * Convenience overload to accept raw float arrays for 1D inputs.
     */
    public Tensor predictThreadSafe(float... input) {
        return predictThreadSafe(Tensor.of(input));
    }
    
    /**
     * Backpropagate a gradient through the network without updating external loss.
     */
    public Tensor backpropagate(Tensor grad, float learningRate) {
        Tensor g = grad;
        for (int i = layers.length - 1; i >= 0; i--) {
            g = layers[i].backward(g, learningRate);
        }
        return g;
    }
    
    /**
     * Train one sample and return the scalar loss value.
     */
    public float trainSingle(LossFunction lossFunction, Tensor x, Tensor y, float learningRate) {
        Tensor output = predict(x);
        float loss = lossFunction.loss(y, output);
        Tensor grad = lossFunction.lossPrime(y, output);
        backpropagate(grad, learningRate);
        return loss;
    }
    
    /**
     * Train one sample and return the scalar loss value.
     */
    public Tensor trainSingleGrad(LossFunction lossFunction, Tensor x, Tensor y, float learningRate) {
        Tensor output = predict(x);
        Tensor grad = lossFunction.lossPrime(y, output);
        return backpropagate(grad, learningRate);
    }
    
    @Override
    public NeuralNetwork clone() {
        try {
            NeuralNetwork cloned = (NeuralNetwork) super.clone();
            Layer[] clonedLayers = new Layer[this.layers.length];
            for (int i = 0; i < this.layers.length; i++) {
                clonedLayers[i] = this.layers[i].clone();
            }
            cloned.layers = clonedLayers;
            return cloned;
        } catch (CloneNotSupportedException e) {
            // Should not happen because we implement Cloneable
            throw new AssertionError(e);
        }
    }
}