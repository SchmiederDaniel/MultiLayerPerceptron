package neuralnetwork.activation;

import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;
import neuralnetwork.layer.Layer;

public abstract class Activation extends Layer {
    protected Tensor lastInput;

    protected abstract Tensor activation(Tensor input);
    protected abstract Tensor activationPrime(Tensor input);

    @Override
    public Tensor forward(Tensor input) {
        Tensor x = input;
        this.lastInput = x.deepCopy();
        return activation(x);
    }

    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        Tensor gradOut = outputGradient;
        Tensor prime = activationPrime(lastInput);
        return gradOut.multiply(prime); // element-wise
    }

    @Override
    public Activation clone() {
        Activation cloned = (Activation) super.clone();
        // Do not share cached inputs across clones
        cloned.lastInput = null;
        return cloned;
    }
}