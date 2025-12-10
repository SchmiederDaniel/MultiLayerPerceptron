package neuralnetwork.lossfunction;

import neuralnetwork.math.Tensor;

import java.util.Arrays;

public class AbsoluteLoss implements LossFunction {
    
    @Override
    public float loss(Tensor yTrue, Tensor yPred) {
        if(!Arrays.equals(yTrue.shape(), yPred.shape()))
            throw new IllegalArgumentException("Shapes don't match " + yTrue.shapeString() + ", " + yPred.shapeString());
        
        Tensor diff = yPred.subtract(yTrue).applyOperation(Math::abs);
//        float sum = 0;
//        for (int rowIndex = 0; rowIndex < yTrue.rows(); rowIndex++) {
//            for (int colIndex = 0; colIndex < yTrue.cols(); colIndex++) {
//                sum += Math.abs(yPred.data[rowIndex][colIndex] - yTrue.data[rowIndex][colIndex]);
//            }
//        }
        float sum = diff.sum();
        return sum / (yTrue.size() * yPred.size());
    }
    
    @Override
    public Tensor lossPrime(Tensor yTrue, Tensor yPred) {
        if(!Arrays.equals(yTrue.shape(), yPred.shape()))
            throw new IllegalArgumentException("Shapes don't match " + yTrue.shapeString() + ", " + yPred.shapeString());
        return yPred.subtract(yTrue);
    }
}