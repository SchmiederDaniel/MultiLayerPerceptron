package neuralnetwork.optimizer;

import neuralnetwork.layer.Dense;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;

public class Adam implements Optimizer {
    private final Dense layer;
    
    // hyperparameters (exposed for tuning if desired)
    private final float beta1;
    private final float beta2;
    private final float eps;
    
    // moment estimates
    private Tensor mW; // Matrix
    private Tensor vW; // Matrix
    private Tensor mB; // Vector
    private Tensor vB; // Vector
    
    private int t = 0;
    
    public Adam(Dense layer) {
        this(layer, 0.9f, 0.999f, 0.00000001f);
    }
    
    public Adam(Dense layer, float beta1, float beta2, float eps) {
        this.layer = layer;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        
        // initialize moment tensors to zeros using the same shapes as parameters
        int[] wshape = layer.W.shape(); // (out, in)
        int[] bshape = layer.b.shape(); // (out)
        this.mW = Tensor.full(wshape, 0f);
        this.vW = Tensor.full(wshape, 0f);
        this.mB = Tensor.full(bshape, 0f);
        this.vB = Tensor.full(bshape, 0f);
    }
    
    @Override
    public void step(float[][] dWarr, float[] dbarr, float learningRate) {
        t += 1;
        
        // Wrap grads as tensors
        Tensor dW = Tensor.of(dWarr); // Matrix
        Tensor db = Tensor.of(dbarr); // Vector
        
        // Cast convenient scalar tensors
        Tensor oneMinusBeta1 = Tensor.of(1f - beta1);
        Tensor oneMinusBeta2 = Tensor.of(1f - beta2);
        
        // m = beta1 * m + (1-beta1) * grad
        mW = mW.multiply(Tensor.of(beta1)).add( dW.multiply(oneMinusBeta1) );
        mB = mB.multiply(Tensor.of(beta1)).add( db.multiply(oneMinusBeta1) );
        
        // v = beta2 * v + (1-beta2) * (grad * grad)
        Tensor dW2 = dW.multiply(dW); // elementwise square
        Tensor db2 = db.multiply(db);
        
        vW = vW.multiply(Tensor.of(beta2)).add( dW2.multiply(oneMinusBeta2) );
        vB = vB.multiply(Tensor.of(beta2)).add( db2.multiply(oneMinusBeta2) );
        
        // compute effective learning rate scalar: lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        float lrTscalar = learningRate * (float) Math.sqrt(1 - Math.pow(beta2, t)) / (1 - (float) Math.pow(beta1, t));
        Tensor lrT = Tensor.of(lrTscalar);
        
        // denom = sqrt(v) + eps  -> do elementwise sqrt via applyOperation
        Tensor vW_sqrt = vW.applyOperation(x -> (float) Math.sqrt(x));
        Tensor vB_sqrt = vB.applyOperation(x -> (float) Math.sqrt(x));
        
        // add epsilon (broadcast scalar)
        Tensor denomW = vW_sqrt.add( Tensor.of(eps) );
        Tensor denomB = vB_sqrt.add( Tensor.of(eps) );
        
        // m / denom
        Tensor stepW = mW.divide(denomW).multiply(lrT); // elementwise scale
        Tensor stepB = mB.divide(denomB).multiply(lrT);
        
        // parameter update: param = param - step
        layer.W = (Matrix) layer.W.subtract(stepW);
        layer.b = (Vector) layer.b.subtract(stepB);
    }
}