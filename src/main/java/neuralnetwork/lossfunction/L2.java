package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

import java.util.Arrays;

public class L2 implements LossFunction {

    @Override
    public float loss(Tensor y_true, Tensor y_pred) {
        if (Tensor.ENABLE_SHAPE_CHECKS) {
            if (!Arrays.equals(y_true.shape(), y_pred.shape()))
                throw new IllegalArgumentException(
                    "Rows or cols doesn't match " +
                        Arrays.toString(y_true.shape()) + ", " +
                        Arrays.toString(y_pred.shape()));
        }

        // diff = prediction - truth
        Tensor diff = y_pred.subtract(y_true);

        // squared = diff * diff (elementwise)
        Tensor squared = diff.multiply(diff);

        // sum all elements
        float[] vals = squared.toFlatArray();
        float sum = 0f;
        for (float v : vals) sum += v;

        // mean squared error
        return sum / vals.length;
    }

    @Override
    public Tensor lossPrime(Tensor y_true, Tensor y_pred) {
        if (Tensor.ENABLE_SHAPE_CHECKS) {
            if (!Arrays.equals(y_true.shape(), y_pred.shape()))
                throw new IllegalArgumentException(
                    "Rows or cols doesn't match " +
                        y_true.shapeString() + ", " + y_pred.shapeString());
        }

        // derivative of (1/N) * sum((y_pred - y_true)^2)
        // is 2*(y_pred - y_true) / N

        Tensor diff = y_pred.subtract(y_true);
        float invN = 2f / diff.size();

        return diff.applyOperation(x -> x * invN);
    }
}