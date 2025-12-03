package neuralnetwork.math.tensor;

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Matrix extends Tensor {
    public final float[][] values;
    
    public Matrix(float[][] values) {
        this.values = values;
    }
    
    @Override
    public Tensor applyOperation(FloatOperator operation) {
        float[][] result = new float[this.values.length][this.values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = operation.applyAsFloat(this.values[i][j]);
            }
        }
        return new Matrix(result);
    }
    
    public Matrix applyVectorOperation(Vector vector, FloatBinaryOperator operation) {
        float[][] result = new float[this.values.length][this.values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = operation.applyAsFloat(this.values[i][j], vector.values[j]);
            }
        }
        return new Matrix(result);
    }
    
    public Matrix applyMatrixOperation(Matrix matrix, FloatBinaryOperator operation) {
        float[][] result = new float[this.values.length][this.values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = operation.applyAsFloat(this.values[i][j], matrix.values[i][j]);
            }
        }
        return new Matrix(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a + ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a + b);
        } else if (tensor instanceof Matrix) {
            return applyMatrixOperation((Matrix) tensor, (a, b) -> a + b);
        } else {
            return tensor.add(this);
        }
    }
    
    @Override
    public Tensor subtract(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a - ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a - b);
        } else if (tensor instanceof Matrix) {
            return applyMatrixOperation((Matrix) tensor, (a, b) -> a - b);
        } else {
            return tensor.subtract(this);
        }
    }
    
    @Override
    public Tensor mul(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a * ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return multiplyVector((Vector) tensor);
        } else if (tensor instanceof Matrix) {
            // Matrix-matrix multiplication
            return multiplyMatrix((Matrix) tensor);
        } else {
            return tensor.mul(this);
        }
    }
    
    /**
     * Applies matrix multiplication with two tensors.
     * 
     * <p>Numpy example:</p>
     * <pre>{@code
     * a = np.array([1], [2], [3])
     * b = np.array([3, 2, 4])
     * a * b = [3 4 12]
     * a.__mul__(b) = [3 4 12]
     * }</pre>
     *
     * @param tensor
     * @return
     */
    @NotNull
    private Matrix multiplyMatrix(Matrix tensor) {
        Matrix matrix = tensor;
        float[][] result = new float[this.values.length][matrix.values[0].length];
        for (int i = 0; i < this.values.length; i++) {
            for (int j = 0; j < matrix.values[0].length; j++) {
                float sum = 0;
                for (int k = 0; k < this.values[0].length; k++) {
                    sum += this.values[i][k] * matrix.values[k][j];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }
    
    /**
     * Applies matrix and vector multiplication.
     *
     * <p>Numpy example:</p>
     * <pre>{@code
     * a = np.array([[1, 2, 3], [3, 2, 4]])
     * b = np.array([1, 2, 3])
     * a * b = [[ 1  4  9], [ 3  4 12]]
     * a.__mul__(b) = [3 4 12]
     * }</pre>
     *
     * @param tensor
     * @return
     */
    @NotNull
    private Vector multiplyVector(Vector tensor) {
        // Matrix-vector multiplication
        Vector vector = tensor;
        float[] result = new float[this.values.length];
        for (int i = 0; i < values.length; i++) {
            float sum = 0;
            for (int j = 0; j < values[0].length; j++) {
                sum += this.values[i][j] * vector.values[j];
            }
            result[i] = sum;
        }
        return getVector(result);
    }
    
    @NotNull
    private static Vector getVector(float[] result) {
        return new Vector(result);
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a * ((Scalar) tensor).value);
        } else if (tensor instanceof Matrix) {
            return applyMatrixOperation((Matrix) tensor, (a, b) -> a * b);
        } else {
            throw new IllegalArgumentException("Hadamard product not supported for Matrix and " + tensor.type());
        }
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a / ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a / b);
        } else if (tensor instanceof Matrix) {
            return applyMatrixOperation((Matrix) tensor, (a, b) -> a / b);
        } else {
            return tensor.divide(this);
        }
    }
    
    @Override
    public Tensor transpose() {
        float[][] result = new float[this.values[0].length][this.values.length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[j][i] = this.values[i][j];
            }
        }
        return new Matrix(result);
    }
    
    @Override
    public Tensor deepCopy() {
        float[][] copyOfValues = Arrays.stream(this.values).map(float[]::clone).toArray(float[][]::new);
        return new Matrix(copyOfValues);
    }
    
    @Override
    public String type() {
        return "Matrix";
    }
    
    @Override
    public String toString() {
        String dimensions = Arrays.stream(this.values)
            .map(e -> String.valueOf(e.length))
            .collect(Collectors.joining(", "));
        return type() + "(" + dimensions + ")";
    }
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Matrix) {
            return Arrays.deepEquals(this.values, ((Matrix) obj).values);
        }
        return false;
    }
}