package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

import java.util.Arrays;

public class MeanSquareError implements LossFunction {
    @Override
    public float loss(Tensor yTrue, Tensor yPred) {
        // Validate sizes (optionally) and work with flattened arrays to support any rank.
        if (Tensor.ENABLE_SHAPE_CHECKS && yTrue.size() != yPred.size()) {
            throw new IllegalArgumentException(
                    "MSE: size mismatch " + Arrays.toString(yTrue.shape()) + " vs " + Arrays.toString(yPred.shape()));
        }
        float[] yt = yTrue.toFlatArray();
        float[] yp = yPred.toFlatArray();
        if (yt.length != yp.length) {
            throw new IllegalArgumentException("MSE: length mismatch " + yt.length + " vs " + yp.length);
        }
        float sum = 0f;
        for (int i = 0; i < yt.length; i++) {
            float d = yp[i] - yt[i];
            sum += d * d;
        }
        return (yt.length == 0) ? 0f : sum / yt.length;
    }

    @Override
    public Tensor lossPrime(Tensor yTrue, Tensor yPred) {
        // Gradient w.r.t. yPred: 2*(yPred - yTrue)/N, preserving the original shape via tensor ops.
        if (Tensor.ENABLE_SHAPE_CHECKS && yTrue.size() != yPred.size()) {
            throw new IllegalArgumentException(
                    "MSE grad: size mismatch " + Arrays.toString(yTrue.shape()) + " vs " + Arrays.toString(yPred.shape()));
        }
        float n = (float) yPred.size();
        if (n == 0f) {
            // Return a zero-like tensor with the same shape
            return yPred.copyFill(0f);
        }
        return yPred.subtract(yTrue).multiply(Tensor.of(2f / n));
    }
}