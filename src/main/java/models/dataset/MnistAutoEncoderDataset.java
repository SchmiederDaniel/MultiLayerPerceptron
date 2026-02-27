package models.dataset;

import neuralnetwork.math.Tensor;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * MNIST dataset for autoencoder training/testing.
 * Inputs are normalized image pixels and targets are identical to inputs (reconstruction).
 * The boolean flag selects between training and test split.
 */
public class MnistAutoEncoderDataset extends DataSet {
    private final boolean train;

    public MnistAutoEncoderDataset(int batchSize, boolean train) {
        super(batchSize, ThreadLocalRandom.current().nextInt());
        this.train = train;
    }

    @Override
    protected Tensor[] inputs() {
        List<int[]> imgs = train ? MNISTLoader.trainData() : MNISTLoader.testData();
        Tensor[] result = new Tensor[imgs.size()];
        for (int i = 0; i < imgs.size(); i++) {
            int[] intValues = imgs.get(i);
            float[] floats = MnistDataset.convertAndNormalize(intValues);
            result[i] = Tensor.of(floats);
        }
        return result;
    }

    @Override
    protected Tensor[] targets() {
        // For autoencoder, targets are the same as inputs
        return inputs();
    }
}