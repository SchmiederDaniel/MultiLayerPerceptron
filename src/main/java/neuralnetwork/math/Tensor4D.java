package neuralnetwork.math;

import java.util.Arrays;

public class Tensor4D extends Tensor {
    private final float[][][][] values;
    
    public Tensor4D(float[][][][] values) {
        this.values = values;
    }
    
    @Override
    public Tensor applyOperation(FloatOperator operation) {
        float[][][][] result = new float
            [this.values.length]
            [this.values[0].length]
            [this.values[0][0].length]
            [this.values[0][0][0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                for (int k = 0; k < values[0][0].length; k++) {
                    for (int l = 0; l < values[0][0][0].length; l++) {
                        result[i][j][k][l] = operation.applyAsFloat(this.values[i][j][k][l]);
                    }
                }
            }
        }
        return new Tensor4D(result);
    }
    
    /**
     * Applies a binary operation between this 4D tensor and a Vector broadcast along the last dimension.
     */
    public Tensor4D applyVectorOperation(Vector vector, FloatBinaryOperator operation) {
        int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
        if (ENABLE_SHAPE_CHECKS && vector.shape()[0] != c)
            throw new IllegalArgumentException("Vector length must match last dim: " + vector.shape()[0] + " vs " + c);
        float[][][][] result = new float[d1][d2][r][c];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int x = 0; x < r; x++) {
                    for (int y = 0; y < c; y++) {
                        result[i][j][x][y] = operation.applyAsFloat(values[i][j][x][y], vector.get(y));
                    }
                }
            }
        }
        return new Tensor4D(result);
    }
    
    /**
     * Applies a binary operation between this 4D tensor and a Matrix broadcast across the first two dimensions.
     */
    public Tensor4D applyMatrixOperation(Matrix matrix, FloatBinaryOperator operation) {
        int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
        if (ENABLE_SHAPE_CHECKS) {
            float[][] mv = matrix.getValues();
            if (mv.length != r || mv[0].length != c)
                throw new IllegalArgumentException("Matrix shape must match last two dims: (" + r + "," + c + ") vs (" + mv.length + "," + mv[0].length + ")");
        }
        float[][][][] result = new float[d1][d2][r][c];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int x = 0; x < r; x++) {
                    for (int y = 0; y < c; y++) {
                        result[i][j][x][y] = operation.applyAsFloat(values[i][j][x][y], matrix.get(x, y));
                    }
                }
            }
        }
        return new Tensor4D(result);
    }
    
    /**
     * Applies a binary operation between this 4D tensor and a 3D tensor broadcast across the first dimension.
     */
    public Tensor4D applyTensor3DOperation(Tensor3D tensor3D, FloatBinaryOperator operation) {
        int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
        if (ENABLE_SHAPE_CHECKS) {
            int[] s = tensor3D.shape();
            if (s[0] != d2 || s[1] != r || s[2] != c)
                throw new IllegalArgumentException("Tensor3D shape must match [d2,r,c]");
        }
        float[][][][] result = new float[d1][d2][r][c];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int x = 0; x < r; x++) {
                    for (int y = 0; y < c; y++) {
                        result[i][j][x][y] = operation.applyAsFloat(values[i][j][x][y], tensor3D.get(j, x, y));
                    }
                }
            }
        }
        return new Tensor4D(result);
    }
    
    /**
     * Applies an element-wise binary operation between two 4D tensors with identical shapes.
     */
    public Tensor4D applyTensor4DOperation(Tensor4D other, FloatBinaryOperator operation) {
        int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
        if (ENABLE_SHAPE_CHECKS) {
            int[] s = other.shape();
            if (s[0] != d1 || s[1] != d2 || s[2] != r || s[3] != c)
                throw new IllegalArgumentException("Tensor4D element-wise requires same shape");
        }
        float[][][][] result = new float[d1][d2][r][c];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int x = 0; x < r; x++) {
                    for (int y = 0; y < c; y++) {
                        result[i][j][x][y] = operation.applyAsFloat(values[i][j][x][y], other.values[i][j][x][y]);
                    }
                }
            }
        }
        return new Tensor4D(result);
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
        } else if (tensor instanceof Tensor4D t4) {
            return applyTensor4DOperation(t4, (a, b) -> a + b);
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
        } else if (tensor instanceof Tensor4D t4) {
            return applyTensor4DOperation(t4, (a, b) -> a - b);
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
        } else if (tensor instanceof Tensor4D t4) {
            return applyTensor4DOperation(t4, (a, b) -> a * b);
        } else {
            throw new IllegalArgumentException("Hadamard product not supported for Tensor4D and " + tensor.type());
        }
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        if (tensor instanceof Vector v) {
            // Batched matvec over last two dims; result dims: [d1][d2][r]
            int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
            if (ENABLE_SHAPE_CHECKS && v.shape()[0] != c)
                throw new IllegalArgumentException("Tensor4D@Vector mismatch: last dim=" + c + " vs vector=" + v.shape()[0]);
            float[][][] result = new float[d1][d2][r];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    for (int x = 0; x < r; x++) {
                        float sum = 0f;
                        for (int y = 0; y < c; y++) {
                            sum += values[i][j][x][y] * v.get(y);
                        }
                        result[i][j][x] = sum;
                    }
                }
            }
            return new Tensor3D(result);
        } else if (tensor instanceof Matrix m) {
            // Batched matmul on last two dims; result dims: [d1][d2][r][n]
            int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
            float[][] mv = m.getValues();
            if (ENABLE_SHAPE_CHECKS && c != mv.length)
                throw new IllegalArgumentException("Tensor4D@Matrix mismatch: inner dim=" + c + " vs m.rows=" + mv.length);
            int n = mv[0].length;
            float[][][][] result = new float[d1][d2][r][n];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    for (int x = 0; x < r; x++) {
                        for (int y = 0; y < n; y++) {
                            float sum = 0f;
                            for (int k = 0; k < c; k++) {
                                sum += values[i][j][x][k] * mv[k][y];
                            }
                            result[i][j][x][y] = sum;
                        }
                    }
                }
            }
            return new Tensor4D(result);
        } else if (tensor instanceof Tensor4D t4) {
            // Batched matmul per [i][j]: (r x c) @ (c x n)
            int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
            int n = t4.shape()[3];
            if (ENABLE_SHAPE_CHECKS) {
                int[] s = t4.shape();
                if (s[0] != d1 || s[1] != d2 || s[2] != c)
                    throw new IllegalArgumentException("Tensor4D@Tensor4D mismatch over last dims");
            }
            float[][][][] result = new float[d1][d2][r][n];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    for (int x = 0; x < r; x++) {
                        for (int y = 0; y < n; y++) {
                            float sum = 0f;
                            for (int k = 0; k < c; k++) {
                                sum += values[i][j][x][k] * t4.get(i, j, k, y);
                            }
                            result[i][j][x][y] = sum;
                        }
                    }
                }
            }
            return new Tensor4D(result);
        } else if (tensor instanceof Scalar) {
            throw new IllegalArgumentException("Tensor4D @ Scalar is not defined in NumPy");
        }
        throw new IllegalArgumentException("Unsupported matmul for Tensor4D with " + tensor.type());
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
        } else if (tensor instanceof Tensor4D t4) {
            return applyTensor4DOperation(t4, (a, b) -> a / b);
        } else {
            return tensor.divide(this);
        }
    }
    
    @Override
    public Tensor transpose() {
        int d1 = values.length, d2 = values[0].length, r = values[0][0].length, c = values[0][0][0].length;
        float[][][][] result = new float[d1][d2][c][r];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int x = 0; x < r; x++) {
                    for (int y = 0; y < c; y++) {
                        result[i][j][y][x] = values[i][j][x][y];
                    }
                }
            }
        }
        return new Tensor4D(result);
    }
    
    @Override
    public Tensor deepCopy() {
        float[][][][] copyOfValues =
            Arrays.stream(values).map(clone3d -> Arrays.stream(clone3d)
                .map(clone2d -> Arrays.stream(clone2d).map(float[]::clone).toArray(float[][]::new))
                .toArray(float[][][]::new)
            ).toArray(float[][][][]::new);
        return new Tensor4D(copyOfValues);
    }
    
    @Override
    public String type() {
        return "Tensor4D";
    }
    
    @Override
    public String toString() {
        return super.toString();
    }
    
    public float[][][][] getValues() {
        return values;
    }
    
    public float get(int i, int j, int k, int l) {
        return values[i][j][k][l];
    }
    
    @Override
    public int[] shape() {
        return new int[] { values.length, values[0].length, values[0][0].length, values[0][0][0].length };
    }
    
    @Override
    public float sum() {
        float sum = 0f;
        for (float[][][] tensor3d : values)
            for (float[][] matrix : tensor3d)
                for (float[] vector : matrix)
                    for (float scalar : vector)
                        sum += scalar;
        return sum;
    }
}