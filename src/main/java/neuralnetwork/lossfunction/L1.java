package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

import java.util.Arrays;

public class L1 implements LossFunction {
    
    @Override
    public float loss(Tensor y_true, Tensor y_pred) {
        if (Tensor.ENABLE_SHAPE_CHECKS) {
            if (!Arrays.equals(y_true.shape(), y_pred.shape()))
                throw new IllegalArgumentException(
                    "Rows or cols doesn't match " +
                        Arrays.toString(y_true.shape()) + ", " +
                        Arrays.toString(y_pred.shape()));
        }
        
        // element-wise absolute difference
        Tensor diff = y_pred.subtract(y_true).applyOperation(Math::abs);
        
        // flatten + sum
        float[] vals = diff.toFlatArray();
        float sum = 0f;
        for (float v : vals) sum += v;
        
        // return mean absolute error
        return sum / vals.length;
    }
    
    @Override
    public Tensor lossPrime(Tensor y_true, Tensor y_pred) {
        if (Tensor.ENABLE_SHAPE_CHECKS) {
            if (!Arrays.equals(y_true.shape(), y_pred.shape()))
                throw new IllegalArgumentException("Rows or cols doesn't match " + y_true.shapeString() + ", " + y_pred.shapeString());
        }
        
        // derivative of |x| is sign(x), with sign(0) = 0
        Tensor diff = y_pred.subtract(y_true);
        
        return diff.applyOperation(x -> {
            if (x > 0) return 1f;
            if (x < 0) return -1f;
            return 0f; // subgradient at zero
        });
    }
}