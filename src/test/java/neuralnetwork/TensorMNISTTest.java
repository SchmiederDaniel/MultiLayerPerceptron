package neuralnetwork;

import neuralnetwork.init.Distribution;
import neuralnetwork.init.SeededDistribution;
import neuralnetwork.math.Tensor;
import neuralnetwork.math.Vector;
import neuralnetwork.activation.Sigmoid;
import neuralnetwork.layer.Dense;
import neuralnetwork.optimizer.OptimizerType;
import org.junit.jupiter.api.Test;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorMNISTTest {
    private static final Distribution distribution = new SeededDistribution(123);

    private static Vector toInputVector(int[] pixels) {
        float[] v = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) v[i] = pixels[i] / 255.0f;
        return new Vector(v);
    }

    private static Vector toOneHot(int label, int classes) {
        float[] y = new float[classes];
        y[label] = 1f;
        return new Vector(y);
    }

    @Test
    void learnsOnSmallMnistSubset() {
        // Build a tiny network: 784 -> 20 -> 10 with sigmoid activations
        Dense l1 = new Dense(28 * 28, 20, distribution, OptimizerType.SGD);
        Sigmoid a1 = new Sigmoid();
        Dense l2 = new Dense(20, 10, distribution, OptimizerType.SGD);
        Sigmoid a2 = new Sigmoid();
        NeuralNetwork net = new NeuralNetwork(l1, a1, l2, a2);
        
        neuralnetwork.lossfunction.LossFunction loss = new neuralnetwork.lossfunction.MeanSquareError();

        // Load a small subset of MNIST (use a test set to avoid large training resource)
        List<int[]> images = readTestImagesSafe();
        List<Integer> labels = readTestLabelsSafe();

        int trainN = Math.min(500, images.size());
        int evalN = Math.min(200, trainN);

        // Initial loss on eval subset
        float initialLoss = 0f;
        for (int i = 0; i < evalN; i++) {
            Vector x = toInputVector(images.get(i));
            Vector y = toOneHot(labels.get(i), 10);
            Tensor pred = net.predict(x);
            initialLoss += loss.loss(y, pred);
        }
        initialLoss /= evalN;

        // Train for a few epochs on the small subset
        float lr = 0.3f;
        int epochs = 2;
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < trainN; i++) {
                Vector x = toInputVector(images.get(i));
                Vector y = toOneHot(labels.get(i), 10);
                net.trainSingle(loss, x, y, lr);
            }
        }

        // Final loss on same eval subset
        float finalLoss = 0f;
        for (int i = 0; i < evalN; i++) {
            Vector x = toInputVector(images.get(i));
            Vector y = toOneHot(labels.get(i), 10);
            Tensor pred = net.predict(x);
            finalLoss += loss.loss(y, pred);
        }
        finalLoss /= evalN;

        // Assert that loss decreased by at least 10%
        assertTrue(finalLoss < initialLoss * 0.9f,
                "Expected training to reduce loss by ~10% on a small subset; initial=" + initialLoss + ", final=" + finalLoss);
    }

    private static List<int[]> readTestImagesSafe() {
        byte[] raw = readResource("testimages.ubyte");
        if (raw.length < 16) throw new RuntimeException("testimages.ubyte too small");
        byte[] imageBytes = new byte[raw.length - 16];
        System.arraycopy(raw, 16, imageBytes, 0, imageBytes.length);
        int pixelsPerImage = 28 * 28;
        int count = imageBytes.length / pixelsPerImage;
        List<int[]> images = new ArrayList<>(count);
        for (int idx = 0; idx < count; idx++) {
            int[] img = new int[pixelsPerImage];
            int base = idx * pixelsPerImage;
            for (int p = 0; p < pixelsPerImage; p++) {
                img[p] = imageBytes[base + p] & 0xff;
            }
            images.add(img);
        }
        return images;
    }

    private static List<Integer> readTestLabelsSafe() {
        byte[] raw = readResource("testlabels.ubyte");
        if (raw.length < 8) throw new RuntimeException("testlabels.ubyte too small");
        byte[] labelBytes = new byte[raw.length - 8];
        System.arraycopy(raw, 8, labelBytes, 0, labelBytes.length);
        List<Integer> labels = new ArrayList<>(labelBytes.length);
        for (byte b : labelBytes) labels.add(b & 0xff);
        return labels;
    }

    private static byte[] readResource(String name) {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(
                TensorMNISTTest.class.getClassLoader().getResourceAsStream(name)))) {
            return in.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}