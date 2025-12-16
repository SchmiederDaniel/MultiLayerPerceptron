package neuralnetwork.layer;

import neuralnetwork.init.Distribution;
import neuralnetwork.math.*;
import neuralnetwork.optimizer.Adam;
import neuralnetwork.optimizer.Optimizer;
import neuralnetwork.optimizer.OptimizerType;
import neuralnetwork.optimizer.SGD;

public class Dense extends LearnableLayer {
    private final OptimizerType optimizerType;
    
    private Vector lastInput;
    
    // optimizer (may be null until set)
    private final Optimizer optimizer;
    
    public Dense(int inSize, int outSize, Distribution distribution, OptimizerType optimizerType) {
        this(initRandom(inSize, outSize, distribution), new Vector(new float[outSize]), optimizerType);
    }
    
    public Dense(Matrix W, Vector b, OptimizerType optimizerType) {
        super(W, b);
        this.W = W;
        this.b = b;
        this.optimizer = createOptimizer(optimizerType);
        this.optimizerType = optimizerType;
    }
    
    private Optimizer createOptimizer(OptimizerType optimizerType) {
        if (optimizerType == OptimizerType.SGD)
            return new SGD(this);
        else if (optimizerType == OptimizerType.Adam)
            return new Adam(this);
        return new SGD(this);
    }
    
    private static Matrix initRandom(int inSize, int outSize, Distribution distribution) {
        distribution.setInputSize(inSize);
        distribution.setOutputSize(outSize);
        float[][] w = new float[outSize][inSize];
        for (int i = 0; i < outSize; i++) {
            for (int j = 0; j < inSize; j++) {
                w[i][j] = distribution.randomWeight();
            }
        }
        return new Matrix(w);
    }
    
    @Override
    public Dense clone() {
        Matrix Wcopy = (Matrix) this.W.deepCopy();
        Vector bcopy = (Vector) this.b.deepCopy();
        Dense cloned = new Dense(Wcopy, bcopy, optimizerType);
        cloned.isTraining = this.isTraining;
        // do not carry lastInput/lastOutput
        // Do not automatically copy/attach optimizer: caller can attach a new optimizer if needed
        return cloned;
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Vector x = (Vector) input;
        this.lastInput = (Vector) x.deepCopy();
        Vector y = (Vector) W.matmul(x);
        y = (Vector) y.add(b);
        return y;
    }
    
    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        Vector gradOut = (Vector) outputGradient; // dL/dy
        
        // dL/dW = (dL/dy) outer x   -> build dW  (out, in)
        float[] g = gradOut.getValues();
        float[] x = lastInput.getValues();
        float[][] dW = new float[W.shape()[0]][W.shape()[1]]; // (out, in)
        for (int i = 0; i < dW.length; i++) {
            float gi = g[i];
            for (int j = 0; j < dW[0].length; j++) {
                dW[i][j] = gi * x[j];
            }
        }
        
        // dL/db = dL/dy (vector)
        float[] db = g;
        
        // delegate parameter update to optimizer
        if (this.optimizer == null) {
            // if no optimizer set, fall back to plain SGD inline update (safe default)
            Tensor dWT = Tensor.of(dW);
            Tensor dbT = Tensor.of(db);
            Tensor lrT = Tensor.of(learningRate);
            
            this.W = (Matrix) this.W.subtract(dWT.multiply(lrT));
            this.b = (Vector) this.b.subtract(dbT.multiply(lrT));
        } else {
            this.optimizer.step(dW, db, learningRate);
        }
        
        // dL/dx = W^T @ (dL/dy)
        return W.transpose().matmul(gradOut);
    }
}