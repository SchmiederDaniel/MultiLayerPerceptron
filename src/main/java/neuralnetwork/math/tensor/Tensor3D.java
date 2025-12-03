package neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Tensor3D extends Tensor {
    public final float[][][] values;
    
    public Tensor3D(float[][][] values) {
        this.values = values;
    }
    
    @Override
    public Tensor applyOperation(FloatOperator operation) {
        float[][][] result = new float
            [this.values.length]
            [this.values[0].length]
            [this.values[0][0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                for (int k = 0; k < values[0][0].length; k++) {
                    result[i][j][k] = operation.applyAsFloat(this.values[i][j][k]);
                }
            }
        }
        return new Tensor3D(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        return null;
    }
    
    @Override
    public Tensor subtract(Tensor tensor) {
        return null;
    }
    
    @Override
    public Tensor mul(Tensor tensor) {
        return null;
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        return null;
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        return null;
    }
    
    @Override
    public Tensor transpose() {
        return null;
    }
    
    @Override
    public Tensor deepCopy() {
        float[][][] copyOfValues =
            Arrays.stream(values)
                .map(clone2d -> Arrays.stream(clone2d).map(float[]::clone).toArray(float[][]::new))
                .toArray(float[][][]::new);
        return new Tensor3D(copyOfValues);
    }
    
    @Override
    public String type() {
        return "Tensor3D";
    }
    
    @Override
    public String toString() {
        String dimensions = Arrays.stream(this.values)
            .map(matrix -> "(" + matrix.length + ", " + (matrix.length > 0 ? matrix[0].length : 0) + ")")
            .collect(Collectors.joining(", "));
        return type() + "(" + dimensions + ")";
    }
}