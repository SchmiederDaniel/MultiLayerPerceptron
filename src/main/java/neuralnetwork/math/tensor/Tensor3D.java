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
    
    /**
     * Applies a binary operation between this 3D tensor and a Vector broadcast along the last dimension.
     */
    public Tensor3D applyVectorOperation(Vector vector, FloatBinaryOperator operation) {
        int d = values.length;
        int r = values[0].length;
        int c = values[0][0].length;
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    result[i][j][k] = operation.applyAsFloat(values[i][j][k], vector.values[k]);
                }
            }
        }
        return new Tensor3D(result);
    }
    
    /**
     * Applies a binary operation between this 3D tensor and a Matrix broadcast across the first dimension.
     */
    public Tensor3D applyMatrixOperation(Matrix matrix, FloatBinaryOperator operation) {
        int d = values.length;
        int r = values[0].length;
        int c = values[0][0].length;
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int row = 0; row < r; row++) {
                for (int col = 0; col < c; col++) {
                    result[i][row][col] = operation.applyAsFloat(values[i][row][col], matrix.values[row][col]);
                }
            }
        }
        return new Tensor3D(result);
    }
    
    /**
     * Applies an element-wise binary operation between two 3D tensors of identical shape.
     */
    public Tensor3D applyTensor3DOperation(Tensor3D other, FloatBinaryOperator operation) {
        int d = values.length;
        int r = values[0].length;
        int c = values[0][0].length;
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    result[i][j][k] = operation.applyAsFloat(values[i][j][k], other.values[i][j][k]);
                }
            }
        }
        return new Tensor3D(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a + s.value);
        } else if (tensor instanceof Vector v) {
            return applyVectorOperation(v, (a, b) -> a + b);
        } else if (tensor instanceof Matrix m) {
            return applyMatrixOperation(m, (a, b) -> a + b);
        } else if (tensor instanceof Tensor3D t3) {
            return applyTensor3DOperation(t3, (a, b) -> a + b);
        } else {
            return tensor.add(this);
        }
    }
    
    @Override
    public Tensor subtract(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a - s.value);
        } else if (tensor instanceof Vector v) {
            return applyVectorOperation(v, (a, b) -> a - b);
        } else if (tensor instanceof Matrix m) {
            return applyMatrixOperation(m, (a, b) -> a - b);
        } else if (tensor instanceof Tensor3D t3) {
            return applyTensor3DOperation(t3, (a, b) -> a - b);
        } else {
            return tensor.subtract(this);
        }
    }
    
    @Override
    public Tensor mul(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a * s.value);
        } else if (tensor instanceof Vector v) {
            // Batched matrix-vector multiplication over first dimension
            int d = values.length;
            int r = values[0].length;
            int c = values[0][0].length;
            float[][] result = new float[d][r];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    float sum = 0f;
                    for (int k = 0; k < c; k++) {
                        sum += values[i][row][k] * v.values[k];
                    }
                    result[i][row] = sum;
                }
            }
            return new Matrix(result);
        } else if (tensor instanceof Matrix m) {
            // Batched matrix-matrix multiplication across first dimension
            int d = values.length;
            int r = values[0].length;
            int c = values[0][0].length;
            int n = m.values[0].length;
            float[][][] result = new float[d][r][n];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    for (int col = 0; col < n; col++) {
                        float sum = 0f;
                        for (int k = 0; k < c; k++) {
                            sum += values[i][row][k] * m.values[k][col];
                        }
                        result[i][row][col] = sum;
                    }
                }
            }
            return new Tensor3D(result);
        } else if (tensor instanceof Tensor3D t3) {
            // Batched matrix-matrix multiplication per slice i: (r x c) @ (c x n)
            int d = values.length;
            int r = values[0].length;
            int c = values[0][0].length;
            int n = t3.values[0][0].length;
            float[][][] result = new float[d][r][n];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    for (int col = 0; col < n; col++) {
                        float sum = 0f;
                        for (int k = 0; k < c; k++) {
                            sum += values[i][row][k] * t3.values[i][k][col];
                        }
                        result[i][row][col] = sum;
                    }
                }
            }
            return new Tensor3D(result);
        } else {
            return tensor.mul(this);
        }
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a * s.value);
        } else if (tensor instanceof Vector v) {
            return applyVectorOperation(v, (a, b) -> a * b);
        } else if (tensor instanceof Matrix m) {
            return applyMatrixOperation(m, (a, b) -> a * b);
        } else if (tensor instanceof Tensor3D t3) {
            return applyTensor3DOperation(t3, (a, b) -> a * b);
        } else {
            throw new IllegalArgumentException("Hadamard product not supported for Tensor3D and " + tensor.type());
        }
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a / s.value);
        } else if (tensor instanceof Vector v) {
            return applyVectorOperation(v, (a, b) -> a / b);
        } else if (tensor instanceof Matrix m) {
            return applyMatrixOperation(m, (a, b) -> a / b);
        } else if (tensor instanceof Tensor3D t3) {
            return applyTensor3DOperation(t3, (a, b) -> a / b);
        } else {
            return tensor.divide(this);
        }
    }
    
    @Override
    public Tensor transpose() {
        int d = values.length;
        int r = values[0].length;
        int c = values[0][0].length;
        float[][][] result = new float[d][c][r];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    result[i][k][j] = values[i][j][k];
                }
            }
        }
        return new Tensor3D(result);
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