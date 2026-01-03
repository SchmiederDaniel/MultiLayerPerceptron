package models.dataset;

import neuralnetwork.math.Tensor;

import java.util.List;

public class MnistDataset extends DataSet {
    private final boolean train;
    
    public MnistDataset(int batchSize, boolean train) {
        super(batchSize);
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
        
        Tensor[] result = new Tensor[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = Tensor.of(list.get(i));
        }
        return result;
    }
}