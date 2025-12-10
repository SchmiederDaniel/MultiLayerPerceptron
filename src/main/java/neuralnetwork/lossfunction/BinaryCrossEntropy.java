package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

public class BinaryCrossEntropy implements LossFunction {

    private static final float EPS = 1e-7f;

    @Override
    public float loss(Tensor yTrue, Tensor yPred) {
        // clip yPred into (EPS, 1-EPS)
        Tensor clippedYP = yPred.applyOperation(v ->
            Math.max(EPS, Math.min(1f - EPS, v))
        );

        Tensor term1 = yTrue.multiply(clippedYP.applyOperation((float v) -> (float) Math.log(v)));
        Tensor term2 = (yTrue.copyFill(1f).subtract(yTrue))
                .multiply(clippedYP.applyOperation((float v) -> (float) Math.log(1f - v)));

        Tensor sum = term1.add(term2);
        float total = sum.sum();             // total negative log-likelihood (but still negative)
        long n = yTrue.size();

        return -total / n;                   // mean BCE (positive)
    }

    @Override
    public Tensor lossPrime(Tensor yTrue, Tensor yPred) {
        // clip yPred to avoid division by zero
        Tensor clipped = yPred.applyOperation(v ->
            Math.max(EPS, Math.min(1f - EPS, v))
        );

        long n = yTrue.size();
        float invN = 1f / n;

        // derivative:
        // dL/dyPred = (yPred - yTrue) / (yPred*(1-yPred)) * (1/N)
        Tensor numerator = clipped.subtract(yTrue);
        Tensor denominator = clipped.multiply(clipped.copyFill(1f).subtract(clipped));  // yPred * (1-yPred)

        return numerator.divide(denominator).applyOperation(v -> v * invN);
    }
}