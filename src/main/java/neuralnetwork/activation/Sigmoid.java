package neuralnetwork.activation;

import neuralnetwork.math.Tensor;

public class Sigmoid extends Activation {

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    protected Tensor activation(Tensor input) {
        return input.applyOperation(operand -> sigmoid(operand));
    }

    @Override
    protected Tensor activationPrime(Tensor input) {
        // s * (1 - s)
        Tensor s = activation(input);
        return s.multiply(s.deepCopy().applyOperation(v -> 1f - v));
    }
}