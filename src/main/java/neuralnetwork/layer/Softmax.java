package neuralnetwork.layer;

import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;

public class Softmax extends Layer {
    private Vector lastOutput; // We store Output (y), not Input, for easier derivative calc

    @Override
    public Tensor forward(Tensor input) {
        Vector x = (Vector) input;
        float[] values = x.getValues();
        float[] output = new float[values.length];

        // 1. Find Max (for numerical stability)
        float max = Float.NEGATIVE_INFINITY;
        for (float v : values) {
            if (v > max) max = v;
        }

        // 2. Exponentiate and Sum
        float sum = 0.0f;
        for (int i = 0; i < values.length; i++) {
            output[i] = (float) Math.exp(values[i] - max);
            sum += output[i];
        }

        // 3. Normalize
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }

        Vector result = new Vector(output);
        this.lastOutput = (Vector) result.deepCopy();
        return result;
    }

    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        // The Jacobian of Softmax is complex:
        // If i == j: S_i * (1 - S_i)
        // If i != j: -S_i * S_j
        //
        // Vectorized form for Gradient w.r.t Input:
        // dL/dx = S * (dL/dy) - S * (S dot dL/dy)
        // where * is element-wise and dot is dot-product.

        Vector gradOut = (Vector) outputGradient;
        Vector s = this.lastOutput;

        // 1. Compute dot product: (S . gradOut)
        float dot = 0.0f;
        float[] sVal = s.getValues();
        float[] gVal = gradOut.getValues();
        
        for (int i = 0; i < sVal.length; i++) {
            dot += sVal[i] * gVal[i];
        }

        // 2. Compute Input Gradient: S_i * (gradOut_i - dot)
        float[] dxValues = new float[sVal.length];
        for (int i = 0; i < sVal.length; i++) {
            dxValues[i] = sVal[i] * (gVal[i] - dot);
        }

        return new Vector(dxValues);
    }

    @Override
    public Softmax clone() {
        // Softmax has no weights to copy
        Softmax clone = new Softmax();
        clone.isTraining = this.isTraining;
        return clone;
    }
}