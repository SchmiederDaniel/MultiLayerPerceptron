package neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Tensor3D extends Tensor {
    private final float[][][] values;
    
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
        if (ENABLE_SHAPE_CHECKS && vector.shape()[0] != c)
            throw new IllegalArgumentException("Vector length must match last dim: " + vector.shape()[0] + " vs " + c);
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    result[i][j][k] = operation.applyAsFloat(values[i][j][k], vector.get(k));
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
        if (ENABLE_SHAPE_CHECKS) {
            float[][] mv = matrix.getValues();
            if (mv.length != r || mv[0].length != c)
                throw new IllegalArgumentException("Matrix shape must match last two dims: (" + r + "," + c + ") vs (" + mv.length + "," + mv[0].length + ")");
        }
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int row = 0; row < r; row++) {
                for (int col = 0; col < c; col++) {
                    result[i][row][col] = operation.applyAsFloat(values[i][row][col], matrix.get(row, col));
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
        if (ENABLE_SHAPE_CHECKS) {
            int[] s = other.shape();
            if (s[0] != d || s[1] != r || s[2] != c)
                throw new IllegalArgumentException("Tensor3D element-wise requires same shape");
        }
        float[][][] result = new float[d][r][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    float[][][] ov = other.getValues();
                    result[i][j][k] = operation.applyAsFloat(values[i][j][k], ov[i][j][k]);
                }
            }
        }
        return new Tensor3D(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a + s.get());
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
            return applyOperation(a -> a - s.get());
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
    public Tensor multiply(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a * s.get());
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
    public Tensor matmul(Tensor tensor) {
        if (tensor instanceof Vector v) {
            // Batched matrix-vector multiplication over first dimension
            int d = values.length;
            int r = values[0].length;
            int c = values[0][0].length;
            if (ENABLE_SHAPE_CHECKS && v.shape()[0] != c)
                throw new IllegalArgumentException("Tensor3D@Vector mismatch: last dim=" + c + " vs vector=" + v.shape()[0]);
            float[][] result = new float[d][r];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    float sum = 0f;
                    for (int k = 0; k < c; k++) {
                        sum += values[i][row][k] * v.get(k);
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
            float[][] mv = m.getValues();
            if (ENABLE_SHAPE_CHECKS && c != mv.length)
                throw new IllegalArgumentException("Tensor3D@Matrix mismatch: inner dim=" + c + " vs m.rows=" + mv.length);
            int n = mv[0].length;
            float[][][] result = new float[d][r][n];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    for (int col = 0; col < n; col++) {
                        float sum = 0f;
                        for (int k = 0; k < c; k++) {
                            sum += values[i][row][k] * mv[k][col];
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
            int[] s = t3.shape();
            if (ENABLE_SHAPE_CHECKS && (s[0] != d || s[1] != c))
                throw new IllegalArgumentException("Tensor3D@Tensor3D mismatch: expected (" + d + "," + c + ",n) but got " + java.util.Arrays.toString(s));
            int n = t3.shape()[2];
            float[][][] result = new float[d][r][n];
            for (int i = 0; i < d; i++) {
                for (int row = 0; row < r; row++) {
                    for (int col = 0; col < n; col++) {
                        float sum = 0f;
                        for (int k = 0; k < c; k++) {
                            float[][][] tv = t3.getValues();
                            sum += values[i][row][k] * tv[i][k][col];
                        }
                        result[i][row][col] = sum;
                    }
                }
            }
            return new Tensor3D(result);
        } else if (tensor instanceof Scalar) {
            throw new IllegalArgumentException("Tensor3D @ Scalar is not defined in NumPy");
        }
        throw new IllegalArgumentException("Unsupported matmul for Tensor3D with " + tensor.type());
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        if (tensor instanceof Scalar s) {
            return applyOperation(a -> a / s.get());
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
    public String toString() { return super.toString(); }

    public float[][][] getValues() { return values; }
    public float get(int i, int j, int k) { return values[i][j][k]; }
    @Override
    public int[] shape() { return new int[] { values.length, values[0].length, values[0][0].length }; }
}