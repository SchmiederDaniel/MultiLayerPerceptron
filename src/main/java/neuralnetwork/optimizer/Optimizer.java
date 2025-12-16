package neuralnetwork.optimizer;


public interface Optimizer {
    /**
     * Apply an optimization step for the layer this optimizer was constructed for.
     *
     * @param dW           gradient for W as a 2D float array shaped (out, in)
     * @param db           gradient for b as a 1D float array shaped (out)
     * @param learningRate scalar learning rate
     */
    void step(float[][] dW, float[] db, float learningRate);
}