package neuralnetwork.optimizer;

import neuralnetwork.layer.Dense;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;

public class SGD implements Optimizer {
    private final Dense layer;
    
    public SGD(Dense layer) {
        this.layer = layer;
    }
    
    @Override
    public void step(float[][] dW, float[] db, float learningRate) {
        // Build tensors from raw arrays
        Tensor dWT = Tensor.of(dW);   // Matrix
        Tensor dbT = Tensor.of(db);   // Vector
        Tensor lrT = Tensor.of(learningRate); // Scalar
        
        // W_new = W - lr * dW  (elementwise scalar broadcast)
        layer.W = (Matrix) layer.W.subtract(dWT.multiply(lrT));
        
        // b_new = b - lr * db
        layer.b = (Vector) layer.b.subtract(dbT.multiply(lrT));
    }
}