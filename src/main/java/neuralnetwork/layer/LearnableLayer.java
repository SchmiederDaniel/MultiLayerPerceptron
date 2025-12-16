package neuralnetwork.layer;

import neuralnetwork.math.Matrix;
import neuralnetwork.math.Vector;

public abstract class LearnableLayer extends Layer {
    public Matrix W;
    public Vector b;
    public LearnableLayer(Matrix W, Vector b) {
        this.W = W;
        this.b = b;
    }
}