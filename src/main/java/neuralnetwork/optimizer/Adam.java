package neuralnetwork.optimizer;

import neuralnetwork.layer.LearnableLayer;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;

public class Adam implements Optimizer {
    // Hyperparameters
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final LearnableLayer layer;
    
    // Mutable raw data arrays (The "Moment" estimates)
    // We store these as primitives to avoid object overhead
    private float[][] mW_data;
    private float[][] vW_data;
    private float[] mB_data;
    private float[] vB_data;
    
    private int t = 0;
    
    public Adam(LearnableLayer layer) {
        this(layer, 0.9f, 0.999f, 1e-8f);
    }
    
    public Adam(LearnableLayer layer, float beta1, float beta2, float eps) {
        this.layer = layer;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        
        // Initialize moment arrays to 0
        // Assuming layer.W contains a float[][] and layer.b contains a float[]
        Matrix W = (Matrix) layer.W;
        int rows = W.getValues().length; // or similar accessor
        int cols = W.getValues()[0].length;
        
        this.mW_data = new float[rows][cols];
        this.vW_data = new float[rows][cols];
        this.mB_data = new float[rows];
        this.vB_data = new float[rows];
    }
    
    @Override
    public void step(float[][] dW, float[] db, float learningRate) {
        if (learningRate == 0)
            return;
        
        Tensor B = layer.b;
        Tensor W = layer.W;
        if (B instanceof Vector b) {
            if (W instanceof Matrix w) {
                t++;
                
                // 1. Pre-calculate the efficient learning rate scalar
                // This moves heavy Math.pow and division operations outside the loops
                float beta1_t = (float) Math.pow(beta1, t);
                float beta2_t = (float) Math.pow(beta2, t);
                float alpha = learningRate * (float) Math.sqrt(1.0 - beta2_t) / (1.0f - beta1_t);
                
                // 2. Pre-calculate constants to avoid repeated subtraction
                float oneMinusBeta1 = 1.0f - beta1;
                float oneMinusBeta2 = 1.0f - beta2;
                
                // 3. Update Weights (Fused Loop)
                // Access raw data directly.
                // Assuming layer.W.getData() returns the backing float[][]
                float[][] wData = w.getValues();
                
                for (int i = 0; i < wData.length; i++) {
                    float[] wRow = wData[i];
                    float[] dWRow = dW[i];
                    float[] mRow = mW_data[i];
                    float[] vRow = vW_data[i];
                    
                    for (int j = 0; j < wRow.length; j++) {
                        float g = dWRow[j];
                        
                        // Update biased first moment estimate (m)
                        // m = beta1 * m + (1-beta1) * g
                        float mVal = beta1 * mRow[j] + oneMinusBeta1 * g;
                        mRow[j] = mVal;
                        
                        // Update biased second raw moment estimate (v)
                        // v = beta2 * v + (1-beta2) * g^2
                        float vVal = beta2 * vRow[j] + oneMinusBeta2 * (g * g);
                        vRow[j] = vVal;
                        
                        // Update parameters directly
                        // w = w - alpha * m / (sqrt(v) + eps)
                        wRow[j] -= alpha * mVal / ((float) Math.sqrt(vVal) + eps);
                    }
                }
                
                // 4. Update Biases (Fused Loop)
                float[] bData = b.getValues();
                
                for (int i = 0; i < bData.length; i++) {
                    float g = db[i];
                    
                    // Update m
                    float mVal = beta1 * mB_data[i] + oneMinusBeta1 * g;
                    mB_data[i] = mVal;
                    
                    // Update v
                    float vVal = beta2 * vB_data[i] + oneMinusBeta2 * (g * g);
                    vB_data[i] = vVal;
                    
                    // Update parameters
                    bData[i] -= alpha * mVal / ((float) Math.sqrt(vVal) + eps);
                }
            } else {
                throw new IllegalArgumentException("Adam only supports Matrix weights");
            }
        }
    }
}
/*
package neuralnetwork.optimizer;

import neuralnetwork.layer.Dense;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Scalar;
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
        Matrix dW = Tensor.of(dWarr); // Matrix
        Vector db = Tensor.of(dbarr); // Vector
        
        // Cast convenient scalar tensors
        Tensor oneMinusBeta1 = Tensor.of(1f - beta1);
        Tensor oneMinusBeta2 = Tensor.of(1f - beta2);
        
        // m = beta1 * m + (1-beta1) * grad
        Scalar beta1Scalar = Tensor.of(beta1);
        mW = mW.multiply(beta1Scalar).add(dW.multiply(oneMinusBeta1));
        mB = mB.multiply(beta1Scalar).add(db.multiply(oneMinusBeta1));
        
        // v = beta2 * v + (1-beta2) * (grad * grad)
//        Tensor dW2 = dW.multiply(dW); // elementwise square
        Tensor dW2 = dW.applyOperation(e -> e * e); // More performant elementwise square
//        Tensor db2 = db.multiply(db);
        Tensor db2 = db.applyOperation(e -> e * e);
        
        Scalar beta2Scalar = Tensor.of(beta2);
        vW = vW.multiply(beta2Scalar).add(dW2.multiply(oneMinusBeta2));
        vB = vB.multiply(beta2Scalar).add(db2.multiply(oneMinusBeta2));
        
        // compute effective learning rate scalar: lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        float lrTscalar = (float) (learningRate * Math.sqrt(1d - Math.pow(beta2, t)) / (1d - Math.pow(beta1, t)));
        Tensor lrT = Tensor.of(lrTscalar);
        
        // denom = sqrt(v) + eps  -> do elementwise sqrt via applyOperation
        Tensor vW_sqrt = vW.applyOperation(x -> (float) Math.sqrt(x));
        Tensor vB_sqrt = vB.applyOperation(x -> (float) Math.sqrt(x));
        
        // add epsilon (broadcast scalar)
        Scalar epsScalar = Tensor.of(eps);
        Tensor denomW = vW_sqrt.add(epsScalar);
        Tensor denomB = vB_sqrt.add(epsScalar);
        
        // m / denom
        Tensor stepW = mW.divide(denomW).multiply(lrT); // elementwise scale
        Tensor stepB = mB.divide(denomB).multiply(lrT);
        
        // parameter update: param = param - step
        layer.W = (Matrix) layer.W.subtract(stepW);
        layer.b = (Vector) layer.b.subtract(stepB);
    }
}
 */