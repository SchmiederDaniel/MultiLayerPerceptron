package neuralnetwork;

import models.dataset.Batch;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.lossfunction.MeanSquareError;
import neuralnetwork.lossfunction.CategoricalCrossEntropy;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;
import neuralnetwork.layer.Layer;
import neuralnetwork.layer.LearnableLayer;
import neuralnetwork.layer.Softmax;
import neuralnetwork.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * A simple feed-forward neural network that uses the new math.tensor API (Vector/Matrix/Tensor).
 */
public class NeuralNetwork implements Cloneable {
    public Layer[] layers;
    
    public NeuralNetwork(Layer... layers) {
        this.layers = layers;
    }
    
    public Tensor predict(Tensor input) {
        Tensor out = input;
        for (Layer layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }
    
    /**
     * Thread-safe prediction: uses a deep-cloned network instance so concurrent
     * training cannot mutate the parameters used for this inference call.
     */
    public Tensor predictThreadSafe(Tensor input) {
        NeuralNetwork copy = this.clone();
        // Ensure inference mode
        for (Layer l : copy.layers) l.isTraining = false;
        return copy.predict(input);
    }
    
    /**
     * Convenience overload to accept raw float arrays for 1D inputs.
     */
    public Tensor predictThreadSafe(float... input) {
        return predictThreadSafe(Tensor.of(input));
    }
    
    /**
     * Backpropagate a gradient through the network without updating external loss.
     */
    public Tensor backpropagate(Tensor grad, float learningRate) {
        Tensor g = grad;
        for (int i = layers.length - 1; i >= 0; i--) {
            g = layers[i].backward(g, learningRate);
        }
        return g;
    }
    
    /**
     * Train one sample and return the scalar loss value.
     */
    public float trainSingle(LossFunction lossFunction, Tensor x, Tensor y, float learningRate) {
        Tensor output = predict(x);
        float loss = lossFunction.loss(y, output);
        Tensor grad = lossFunction.lossPrime(y, output);
        backpropagate(grad, learningRate);
        return loss;
    }
    
    /**
     * Train one sample and return the scalar loss value.
     */
    public Tensor trainSingleGrad(LossFunction lossFunction, Tensor x, Tensor y, float learningRate) {
        Tensor output = predict(x);
        Tensor grad = lossFunction.lossPrime(y, output);
        return backpropagate(grad, learningRate);
    }
    
    @Override
    public NeuralNetwork clone() {
        try {
            NeuralNetwork cloned = (NeuralNetwork) super.clone();
            Layer[] clonedLayers = new Layer[this.layers.length];
            for (int i = 0; i < this.layers.length; i++) {
                clonedLayers[i] = this.layers[i].clone();
            }
            cloned.layers = clonedLayers;
            return cloned;
        } catch (CloneNotSupportedException e) {
            // Should not happen because we implement Cloneable
            throw new AssertionError(e);
        }
    }
    
    public void train(Tensor[] batchInputs, Tensor[] batchTargets, float learningRate) {
        if (batchInputs == null || batchTargets == null)
            throw new IllegalArgumentException("Batch or batch contents are null");
        int n = batchInputs.length;
        if (n == 0) return;

        // 1) Decide loss function: Softmax output -> use CategoricalCrossEntropy (expects probabilities), else MSE
        LossFunction loss = (layers.length > 0 && layers[layers.length - 1] instanceof Softmax)
            ? new CategoricalCrossEntropy()
            : new MeanSquareError();

        // 2) Prepare list of learnable layer indices to preserve order
        List<Integer> learnableIdx = new ArrayList<>();
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof LearnableLayer) learnableIdx.add(i);
        }
        final int L = learnableIdx.size();
        if (L == 0) return; // nothing to update

        // 3) Run each sample on its own cloned network to compute gradients (CPU bound)
        int threads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, n));
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<float[][][]>> futures = new ArrayList<>(n);

        // We return a 3D array per sample: [2][layer] -> 0: dW(flat[][] collapsed via null-guard), 1: db(1D)
        // But Java arrays need consistent typing; we'll return an array where index 0 holds dW list packed after,
        // so instead create a custom packing: float[][][] where for each layer k: dW at even index (2*k), db as 1xN at odd index (2*k+1)

        for (int i = 0; i < n; i++) {
            final Tensor x = batchInputs[i];
            final Tensor y = batchTargets[i];
            futures.add(pool.submit(() -> {
                NeuralNetwork copy = this.clone();
                // enable training and accumulation on learnable layers
                for (Layer layer : copy.layers) {
                    layer.isTraining = true;
                    if (layer instanceof LearnableLayer ll) {
                        ll.setAccumulateOnly(true);
                    }
                }
                // forward
                Tensor out = copy.predict(x);
                // grad wrt output
                Tensor grad = loss.lossPrime(y, out);
                // backprop (learningRate is ignored in accumulate-only mode)
                copy.backpropagate(grad, learningRate);

                // collect gradients in order of learnableIdx
                // We'll pack as [2*L] entries: for each k: even index is dW (as 2D), odd index is db as 2D row vector [1][N]
                float[][][] packed = new float[2 * L][][];
                int pos = 0;
                for (int idx : learnableIdx) {
                    LearnableLayer ll = (LearnableLayer) copy.layers[idx];
                    float[][] dW = ll.getLastDW();
                    float[] db = ll.getLastDb();
                    packed[pos++] = dW; // 2D
                    packed[pos++] = new float[][] { db != null ? db : new float[0] }; // store biases as 2D 1xN
                }
                return packed;
            }));
        }

        // 4) Initialize accumulators from current layer shapes using Tensor (Matrix/Vector)
        Matrix[] sum_dW = new Matrix[L];
        Vector[] sum_db = new Vector[L];
        for (int li = 0; li < L; li++) {
            LearnableLayer ll = (LearnableLayer) layers[learnableIdx.get(li)];
            int out = ll.W.shape()[0];
            int in = ll.W.shape()[1];
            sum_dW[li] = new Matrix(new float[out][in]);
            sum_db[li] = new Vector(new float[out]);
        }

        // 5) Aggregate gradients from futures
        try {
            for (Future<float[][][]> f : futures) {
                float[][][] packed = f.get();
                int pos = 0;
                for (int li = 0; li < L; li++) {
                    float[][] dW = packed[pos++];
                    float[][] db2d = packed[pos++];
                    float[] db = (db2d.length > 0) ? db2d[0] : null;

                    if (dW != null) {
                        sum_dW[li] = (Matrix) sum_dW[li]
                                .add(new Matrix(dW));
                    }
                    if (db != null) {
                        sum_db[li] = (Vector) sum_db[li]
                                .add(new Vector(db));
                    }
                }
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        } catch (ExecutionException ee) {
            throw new RuntimeException("Error during parallel batch training", ee);
        } finally {
            pool.shutdown();
        }

        // 6) Average and apply using each layer's optimizer (still uses layer APIs)
        float inv = 1f / n;
        for (int li = 0; li < L; li++) {
            Matrix dWm = (Matrix) sum_dW[li]
                    .multiply(Tensor.of(inv));
            Vector dbv = (Vector) sum_db[li]
                    .multiply(Tensor.of(inv));

            LearnableLayer ll = (LearnableLayer) layers[learnableIdx.get(li)];
            ll.applyGradients(dWm.getValues(), dbv.getValues(), learningRate);
        }
    }
}