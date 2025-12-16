package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;

public class CategoricalCrossEntropy implements LossFunction {
    private static final float EPSILON = 1e-9f;

    @Override
    public float loss(Tensor yTrue, Tensor yPred) {
        // Loss = - Sum(y_true * log(y_pred))
        float[] t = ((Vector) yTrue).getValues();
        float[] p = ((Vector) yPred).getValues();
        
        float sumScore = 0.0f;
        for (int i = 0; i < t.length; i++) {
            // Clip value to avoid log(0)
            float val = Math.max(p[i], EPSILON); 
            sumScore += t[i] * (float)Math.log(val);
        }
        return -sumScore;
    }

    @Override
    public Tensor lossPrime(Tensor yTrue, Tensor yPred) {
        // Gradient = - y_true / y_pred
        float[] t = ((Vector) yTrue).getValues();
        float[] p = ((Vector) yPred).getValues();
        float[] grad = new float[t.length];

        for (int i = 0; i < t.length; i++) {
            float val = Math.max(p[i], EPSILON);
            grad[i] = -t[i] / val;
        }
        
        return new Vector(grad);
    }
}