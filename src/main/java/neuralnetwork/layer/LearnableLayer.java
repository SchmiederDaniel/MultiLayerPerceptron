package neuralnetwork.layer;

import neuralnetwork.math.Matrix;
import neuralnetwork.math.Vector;

public abstract class LearnableLayer extends Layer {
    public Matrix W;
    public Vector b;
    // Last computed parameter gradients (set during backward pass)
    private float[][] lastDW;
    private float[] lastDb;
    // If true, backward() computes and stores gradients but does NOT update parameters.
    private volatile boolean accumulateOnly = false;
    public LearnableLayer(Matrix W, Vector b) {
        this.W = W;
        this.b = b;
    }

    /**
     * Store the last computed gradients for this layer (used for batch accumulation).
     */
    public void setLastGradients(float[][] dW, float[] db) {
        this.lastDW = dW;
        this.lastDb = db;
    }

    /**
     * Retrieve the most recent weight gradient (may be null if backward not yet called).
     */
    public float[][] getLastDW() {
        return lastDW;
    }

    /**
     * Retrieve the most recent bias gradient (may be null if backward not yet called).
     */
    public float[] getLastDb() {
        return lastDb;
    }

    /**
     * Apply gradients to parameters. Subclasses with dedicated optimizers should override
     * to route through their optimizer. Default falls back to plain SGD update.
     */
    public void applyGradients(float[][] dW, float[] db, float learningRate) {
        if (dW != null) {
            this.W = (Matrix) this.W.subtract(new Matrix(dW).multiply(neuralnetwork.math.Tensor.of(learningRate)));
        }
        if (db != null) {
            this.b = (Vector) this.b.subtract(new Vector(db).multiply(neuralnetwork.math.Tensor.of(learningRate)));
        }
    }

    public boolean isAccumulateOnly() {
        return accumulateOnly;
    }

    public void setAccumulateOnly(boolean accumulateOnly) {
        this.accumulateOnly = accumulateOnly;
    }
}