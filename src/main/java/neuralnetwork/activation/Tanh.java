package neuralnetwork.activation;

import neuralnetwork.math.Tensor;

public class Tanh extends Activation {
    @Override
    protected Tensor activation(Tensor input) {
        return input.applyOperation(e -> (float) Math.tanh(e));
    }
    
    @Override
    protected Tensor activationPrime(Tensor input) {
        return input.applyOperation(e -> (float) (1f - (Math.pow((float) Math.tanh(e), 2))));
    }
}
