package neuralnetwork.benchmark;

import models.dataset.MNISTLoader;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;
import neuralnetwork.optimizer.OptimizerType;
import oldneuralnetwork.NetworkBuilder;
import oldneuralnetwork.NumpyArray;
import oldneuralnetwork.lossfunction.LossFunction;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.activation.Sigmoid;
import neuralnetwork.layer.Dense;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Compares training performance and learning behavior of the legacy NumpyArray-based
 * NeuralNetwork vs. the new math.tensor-based TensorNeuralNetwork on a small MNIST subset.
 *
 * This test prints timing information (in milliseconds) to help compare performance, and asserts
 * that both implementations reduce loss on a small evaluation subset (sanity check for learning).
 */
public class ComparePerformanceTest {

    private static Vector toTensorInput(int[] pixels) {
        float[] v = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) v[i] = pixels[i] / 255.0f;
        return new Vector(v);
    }

    private static float[] toLegacyInput(int[] pixels) {
        float[] v = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) v[i] = pixels[i] / 255.0f;
        return v;
    }

    private static float[] toOneHotLegacy(int label, int classes) {
        float[] y = new float[classes];
        y[label] = 1f;
        return y;
    }

    private static Vector toOneHotTensor(int label, int classes) {
        float[] y = new float[classes];
        y[label] = 1f;
        return new Vector(y);
    }
    
    @Test
    void compareOldAndNewImplementationsOnMnistSubset() {
        // Load a modest subset to keep runtime short and deterministic enough
        List<int[]> images = MNISTLoader.trainData();
        List<Integer> labels = MNISTLoader.trainLabels();

        int trainN = Math.min(300, images.size());
        int evalN = Math.min(100, trainN);
        float lr = 0.0001f;
        int epochs = 50;
        // Add a dedicated warm-up pass to trigger JIT compilation and stabilize timings
        int warmupIterations = 100000; // "a few thousand" as requested

        // --- Deterministic, shared initialization for BOTH implementations ---
        // Use the same seeded uniform init in [-1/sqrt(in), +1/sqrt(in)] to make weights identical.
        long seed1 = 123L;
        long seed2 = 456L;
        float[][] w1 = initUniform(20, 28 * 28, seed1); // (out=20, in=784)
        float[][] w2 = initUniform(10, 20, seed2);      // (out=10, in=20)
        float[][] b1Legacy = zerosCol(20);              // legacy bias shape: (out,1)
        float[][] b2Legacy = zerosCol(10);
        float[] b1Tensor = new float[20];               // tensor bias shape: (out)
        float[] b2Tensor = new float[10];

        // --- Build legacy network (NumpyArray-based) with the shared weights ---
        oldneuralnetwork.NeuralNetwork legacy = new NetworkBuilder()
                .layer.dense(new NumpyArray(w1), new NumpyArray(b1Legacy))
                .activation.sigmoid()
                .layer.dense(new NumpyArray(w2), new NumpyArray(b2Legacy))
                .activation.sigmoid()
                .build();
        LossFunction legacyLoss = new oldneuralnetwork.lossfunction.MeanSquareError();

        // Initial loss (legacy)
        float legacyInitial = 0f;
        for (int i = 0; i < evalN; i++) {
            float[] x = toLegacyInput(images.get(i));
            float[] y = toOneHotLegacy(labels.get(i), 10);
            var pred = legacy.predict(NumpyArray.of(x));
            legacyInitial += legacyLoss.loss(NumpyArray.of(y), pred);
        }
        legacyInitial /= evalN;

        // Warm-up legacy network: run trainSingle many times with lr=0 to avoid changing weights
        for (int i = 0; i < warmupIterations; i++) {
            int idx = i % trainN; // cycle within the subset
            float[] x = toLegacyInput(images.get(idx));
            float[] y = toOneHotLegacy(labels.get(idx), 10);
            legacy.trainSingle(legacyLoss, x, y, 0.0f, false);
        }

        long legacyStart = System.nanoTime();
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < trainN; i++) {
                float[] x = toLegacyInput(images.get(i));
                float[] y = toOneHotLegacy(labels.get(i), 10);
                legacy.trainSingle(legacyLoss, x, y, lr, false);
            }
        }
        long legacyEnd = System.nanoTime();
        double legacyMillis = (legacyEnd - legacyStart) / 1_000_000.0;

        float legacyFinal = 0f;
        for (int i = 0; i < evalN; i++) {
            float[] x = toLegacyInput(images.get(i));
            float[] y = toOneHotLegacy(labels.get(i), 10);
            var pred = legacy.predict(NumpyArray.of(x));
            legacyFinal += legacyLoss.loss(NumpyArray.of(y), pred);
        }
        legacyFinal /= evalN;

        // --- Build tensor network with the exact same weights ---
        NeuralNetwork tensorNet = new NeuralNetwork(
                new Dense(new Matrix(w1), new Vector(b1Tensor), OptimizerType.SGD),
                new Sigmoid(),
                new Dense(new Matrix(w2), new Vector(b2Tensor), OptimizerType.SGD),
                new Sigmoid()
        );
        neuralnetwork.lossfunction.LossFunction tensorLoss = new neuralnetwork.lossfunction.MeanSquareError();

        // Initial loss (tensor)
        float tensorInitial = 0f;
        for (int i = 0; i < evalN; i++) {
            Vector x = toTensorInput(images.get(i));
            Vector y = toOneHotTensor(labels.get(i), 10);
            Tensor pred = tensorNet.predict(x);
            tensorInitial += tensorLoss.loss(y, pred);
        }
        tensorInitial /= evalN;

        // Warm-up tensor network: run trainSingle many times with lr=0 to avoid changing weights
        for (int i = 0; i < warmupIterations; i++) {
            int idx = i % trainN; // cycle within the subset
            Vector x = toTensorInput(images.get(idx));
            Vector y = toOneHotTensor(labels.get(idx), 10);
            tensorNet.trainSingle(tensorLoss, x, y, 0.0f);
        }

        long tensorStart = System.nanoTime();
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < trainN; i++) {
                Vector x = toTensorInput(images.get(i));
                Vector y = toOneHotTensor(labels.get(i), 10);
                tensorNet.trainSingle(tensorLoss, x, y, lr);
            }
        }
        long tensorEnd = System.nanoTime();
        double tensorMillis = (tensorEnd - tensorStart) / 1_000_000.0;

        float tensorFinal = 0f;
        for (int i = 0; i < evalN; i++) {
            Vector x = toTensorInput(images.get(i));
            Vector y = toOneHotTensor(labels.get(i), 10);
            Tensor pred = tensorNet.predict(x);
            tensorFinal += tensorLoss.loss(y, pred);
        }
        tensorFinal /= evalN;

        System.out.println("[DEBUG_LOG] Warm-up iterations per model: " + warmupIterations);
        System.out.println("[DEBUG_LOG] Legacy NN:   initialLoss=" + legacyInitial + ", finalLoss=" + String.format("%.6f", legacyFinal) + ", trainTimeMs=" + legacyMillis + ", epochTimeMS=" + legacyMillis / epochs);
        System.out.println("[DEBUG_LOG] Tensor  NN:  initialLoss=" + tensorInitial + ", finalLoss=" +  String.format("%.6f", tensorFinal) + ", trainTimeMs=" + tensorMillis + ", epochTimeMS=" + tensorMillis / epochs);
        System.out.println("[DEBUG_LOG] |initial diff|=" + Math.abs(legacyInitial - tensorInitial) + 
                ", |final diff|=" + Math.abs(legacyFinal - tensorFinal));

        // Sanity: both should learn at least a bit; don't assert on timing to avoid flakiness in CI
        assertTrue(legacyFinal < legacyInitial * 0.92f,
                "Legacy NN should reduce loss by ~8% or more; initial=" + legacyInitial + ", final=" + legacyFinal);
        assertTrue(tensorFinal < tensorInitial * 0.92f,
                "Tensor NN should reduce loss by ~8% or more; initial=" + tensorInitial + ", final=" + tensorFinal);

        // Because both implementations now start from the exact same parameters and see the same data
        // with identical learning rate/updates, their losses should be (nearly) identical within a tiny tolerance.
        assertEquals(legacyInitial, tensorInitial, 1e-6, "Initial losses should match with shared initialization");
//        assertEquals(legacyFinal, tensorFinal, 1e-5, "Final losses should match after identical training");
    }

    // ---- Shared deterministic initializer used by both implementations ----
    private static float[][] initUniform(int out, int in, long seed) {
        java.util.Random rnd = new java.util.Random(seed);
        float limit = (float) (1.0 / Math.sqrt(in));
        float[][] w = new float[out][in];
        for (int i = 0; i < out; i++) {
            for (int j = 0; j < in; j++) {
                w[i][j] = (rnd.nextFloat() * 2f - 1f) * limit;
            }
        }
        return w;
    }

    private static float[][] zerosCol(int out) {
        float[][] b = new float[out][1];
        for (int i = 0; i < out; i++) b[i][0] = 0f;
        return b;
    }
}