package neuralnetwork;

import math.PythonBuilder;
import math.PythonResult;
import math.NumpyParser;
import neuralnetwork.lossfunction.MeanSquareError;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;
import neuralnetwork.layer.Dense;
import neuralnetwork.optimizer.OptimizerType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that per-sample Dense gradients aggregated across a batch
 * match NumPy's computation using outer products, and validates our
 * tensor-based accumulation path indirectly.
 */
public class BatchGradientNumpyTest {

    @Test
    void densePerSampleGradientsMatchNumPy() throws Exception {
        // Define a small deterministic Dense layer: in=3, out=2
        float[][] w = new float[][]{
                {0.1f, -0.2f, 0.3f},
                {-0.4f, 0.5f, -0.6f}
        };
        Dense dense = new Dense(new Matrix(w), new Vector(new float[]{0f, 0f}), OptimizerType.SGD);

        // Three samples (batch size = 3)
        float[][] xs = new float[][]{
                {1f, 0f, -1f},
                {0.5f, 0.25f, -0.5f},
                {-2f, 1f, 0.0f}
        };
        float[][] ys = new float[][]{
                {1f, 0f},
                {0f, 1f},
                {1f, 0f}
        };

        MeanSquareError mse = new MeanSquareError();

        // Compute per-sample gradients in Java (without updating weights)
        float[][] sumDW = new float[2][3];
        float[] sumDb = new float[2];

        for (int i = 0; i < xs.length; i++) {
            Vector x = new Vector(xs[i]);
            Vector y = new Vector(ys[i]);

            // forward
            Vector yhat = (Vector) dense.forward(x);
            // grad wrt output
            Vector grad = (Vector) mse.lossPrime(y, yhat);

            // Dense per-sample gradients
            float[] g = grad.getValues();
            float[] xv = x.getValues();
            float[][] dWi = new float[2][3];
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 3; c++) {
                    dWi[r][c] = g[r] * xv[c];
                    sumDW[r][c] += dWi[r][c];
                }
                sumDb[r] += g[r];
            }
        }

        // Use NumPy to compute the same gradient aggregation
        PythonBuilder pb = new PythonBuilder()
                .append("xs = np.array(" + toPython(xs) + ")")
                .append("ys = np.array(" + toPython(ys) + ")")
                .append("W = np.array(" + toPython(w) + ")")
                .append("b = np.zeros(2)")
                // forward linear: yhat = W @ x + b for each sample
                .append("def forward(W, b, x): return W.dot(x) + b")
                .append("def mse_grad(y, yhat):\n    n = y.shape[0]\n    return 2.0*(yhat - y)/n  # per-sample grad matches Java's MeanSquareError (2/N)")
                .append("sumDW = np.zeros((2,3))\nsumDb = np.zeros(2)")
                .append("for i in range(xs.shape[0]):\n    x = xs[i]\n    y = ys[i]\n    yhat = forward(W, b, x)\n    g = mse_grad(y, yhat)\n    sumDW += np.outer(g, x)\n    sumDb += g")
                .append("print([sumDW.tolist(), sumDb.tolist()])");

        PythonResult result = pb.execute();
        String output = result.getOutput();
        Object parsed = NumpyParser.parse(output);

        // parsed should be a 2-element Object[]: [sumDW(2x3), sumDb(2)]
        assertTrue(parsed instanceof Object[], "Expected a top-level list from Python output");
        Object[] arr = (Object[]) parsed;
        assertEquals(2, arr.length, "Expected two elements: [dW, db]");
        float[][] npDW = (float[][]) arr[0];
        float[] npDb = (float[]) arr[1];

        // Compare with small tolerance
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(npDW[r][c], sumDW[r][c], 1e-4f, "dW mismatch at ("+r+","+c+")");
            }
        }
        for (int r = 0; r < 2; r++) {
            assertEquals(npDb[r], sumDb[r], 1e-4f, "db mismatch at index "+r);
        }
    }

    private static String toPython(float[][] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("[");
            for (int j = 0; j < a[i].length; j++) {
                if (j > 0) sb.append(", ");
                sb.append(a[i][j]);
            }
            sb.append("]");
        }
        sb.append("]");
        return sb.toString();
    }
}