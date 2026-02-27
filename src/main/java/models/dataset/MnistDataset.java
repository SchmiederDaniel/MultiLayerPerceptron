package models.dataset;

import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.math.Tensor;

import java.util.List;

public class MnistDataset extends DataSet {
    private final boolean train;
    
    public MnistDataset(int batchSize, boolean train, int seed) {
        super(batchSize, seed);
        this.train = train;
    }
    
    @Override
    protected Tensor[] inputs() {
        List<int[]> list;
        if (train) {
            list = MNISTLoader.trainData();
        } else {
            list = MNISTLoader.testData();
        }
        
        Tensor[] result = new Tensor[list.size()];
        for (int i = 0; i < list.size(); i++) {
            int[] intValues = list.get(i);
            float[] floats = convertAndNormalize(intValues);
            result[i] = Tensor.of(floats);
        }
        return result;
    }
    
    public static float[] convertAndNormalize(int[] input) {
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i] / 255f;
        }
        return result;
    }
    
    @Override
    protected Tensor[] targets() {
        List<Integer> list;
        if (train) {
            list = MNISTLoader.trainLabels();
        } else {
            list = MNISTLoader.testLabels();
        }
        
        // Return one-hot vectors (10 classes) instead of scalar class indices.
        Tensor[] result = new Tensor[list.size()];
        for (int i = 0; i < list.size(); i++) {
            int label = list.get(i);
            float[] oneHot = new float[10];
            if (label >= 0 && label < 10) oneHot[label] = 1f;
            result[i] = Tensor.of(oneHot);
        }
        return result;
    }
    
    class ExampleUsage {
        public static void main(String[] args) {
            NeuralNetwork network = new NetworkBuilder()
                .layer.dense(2, 1)
                .activation.sigmoid()
                .layer.dense(1, 1)
                .activation.sigmoid()
                .build();
            DataSet trainSet = new MnistDataset(30, true, 0);
            float learningRate = 0.0001f;
            for (Batch batch : trainSet) {
                network.train(batch.inputs, batch.targets, learningRate);
            }
        }
    }
}