package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

public interface LossFunction {
    float loss(Tensor yTrue, Tensor yPred);
    Tensor lossPrime(Tensor yTrue, Tensor yPred);
}