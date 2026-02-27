package neuralnetwork.math;

import java.util.Arrays;


public abstract class Tensor {
    /**
     * Enable or disable shape validation in runtime. When true, operations check
     * shape compatibility and throw {@link IllegalArgumentException} on mismatch.
     * Keep false for maximal performance.
     */
    public static boolean ENABLE_SHAPE_CHECKS = true;
    
    /**
     * Apply a unary operation element‑wise and return a new tensor (no in‑place).
     */
    public abstract Tensor applyOperation(FloatOperator operation);
    
    /**
     * Add two tensors (with broadcasting when supported).
     */
    public abstract Tensor add(Tensor tensor);
    
    /**
     * Subtract tensors (this - tensor) with broadcasting when supported.
     */
    public abstract Tensor subtract(Tensor tensor);
    
    /**
     * Element‑wise (Hadamard) multiplication.
     * <p>
     * NumPy equivalence:
     * - Python: `c = a * b`
     * - Java: `c = a.multiply(b)`
     * <p>
     * Shapes: must be equal or broadcastable (depending on implementation). Returns a new tensor.
     */
    public abstract Tensor multiply(Tensor tensor);
    
    /**
     * Matrix/tensor multiplication (linear algebra over the last dims).
     * <p>
     * NumPy equivalence:
     * - Python: `c = a @ b` or `np.matmul(a, b)`
     * - Java: `c = a.matmul(b)`
     * <p>
     * Examples:
     * - Vector/matrix: `np.array([[1,2,3]]) @ np.array([[1],[2],[3]])`
     * - Matrix/matrix: `np.array([[1,2],[3,4]]) @ np.array([[5,6],[7,8]])`
     * - Batched (Tensor3D/4D): multiplication applies to the last two dims for each batch index.
     */
    public abstract Tensor matmul(Tensor tensor);
    
    /**
     * Divide tensors (this / tensor) with broadcasting when supported.
     */
    public abstract Tensor divide(Tensor tensor);
    
    /**
     * Fill a copy of this tensor's shape with a constant value.
     */
    public Tensor copyFill(float value) {
        return this.applyOperation(_ -> value);
    }
    
    /**
     * Transpose: swap last two axes (for Matrix/Tensor3D/Tensor4D).
     */
    public abstract Tensor transpose();
    
    /**
     * Deep copy (no shared backing arrays).
     */
    public abstract Tensor deepCopy();
    
    /**
     * Type name (Scalar, Vector, Matrix, Tensor3D, Tensor4D).
     */
    public abstract String type();
    
    /**
     * Return the NumPy‑style shape.
     */
    public abstract int[] shape();
    
    /**
     * Return the NumPy‑style shape as String.
     */
    public String shapeString() {
        return Arrays.toString(shape());
    }
    
    /**
     * Rank/ndim (length of shape array).
     */
    public int rank() {
        return shape().length;
    }
    
    /**
     * Total number of elements.
     */
    public long size() {
        int[] s = shape();
        long n = 1L;
        for (int v : s) n *= v;
        return n;
    }
    
    // ---- Factory helpers (non in‑place) ----
    public static Scalar of(double value) {
        return new Scalar((float) value);
    }
    
    public static Vector of(double... values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
        return new Vector(result);
    }
    
    public static Scalar of(float value) {
        return new Scalar(value);
    }
    
    public static Vector of(float... values) {
        return new Vector(values);
    }
    
    public static Matrix of(float[][] values) {
        return new Matrix(values);
    }
    
    public static Tensor3D of(float[][][] values) {
        return new Tensor3D(values);
    }
    
    public static Tensor4D of(float[][][][] values) {
        return new Tensor4D(values);
    }
    
    /**
     * Identity matrix of size n.
     */
    public static Matrix eye(int n) {
        float[][] v = new float[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1f;
        return new Matrix(v);
    }
    
    /**
     * arange(stop): 0,1,2,...,stop-1
     */
    public static Vector arange(int stop) {
        float[] v = new float[stop];
        for (int i = 0; i < stop; i++) v[i] = i;
        return new Vector(v);
    }
    
    /**
     * arange(start, stop, step) similar to NumPy (stop exclusive).
     */
    public static Vector arange(float start, float stop, float step) {
        if (step == 0f) throw new IllegalArgumentException("step must not be 0");
        int n = (int) Math.max(0, Math.ceil((stop - start) / step));
        float[] v = new float[n];
        float cur = start;
        for (int i = 0; i < n; i++, cur += step) v[i] = cur;
        return new Vector(v);
    }
    
    /**
     * full(shape,value) up to rank 4; shape.length==0 returns Scalar.
     */
    public static Tensor full(int[] shape, float value) {
        switch (shape.length) {
            case 0:
                return new Scalar(value);
            case 1: {
                float[] a = new float[shape[0]];
                Arrays.fill(a, value);
                return new Vector(a);
            }
            case 2: {
                float[][] a = new float[shape[0]][shape[1]];
                for (int i = 0; i < shape[0]; i++) Arrays.fill(a[i], value);
                return new Matrix(a);
            }
            case 3: {
                float[][][] a = new float[shape[0]][shape[1]][shape[2]];
                for (int i = 0; i < shape[0]; i++)
                    for (int j = 0; j < shape[1]; j++)
                        Arrays.fill(a[i][j], value);
                return new Tensor3D(a);
            }
            case 4: {
                float[][][][] a = new float[shape[0]][shape[1]][shape[2]][shape[3]];
                for (int i = 0; i < shape[0]; i++)
                    for (int j = 0; j < shape[1]; j++)
                        for (int k = 0; k < shape[2]; k++)
                            Arrays.fill(a[i][j][k], value);
                return new Tensor4D(a);
            }
            default:
                throw new IllegalArgumentException("Only ranks 0..4 supported in full()");
        }
    }
    
    /**
     * Flatten (copy) to a 1D Vector in row‑major order.
     */
    public Vector flatten() {
        float[] out = new float[(int) size()];
        // Default implementation: rely on transpose + getters in subclasses
        // Subclasses may override for efficiency.
        int idx = 0;
        int[] s = shape();
        if (s.length == 0) {
            out[idx++] = ((Scalar) this).get();
        } else if (s.length == 1) {
            float[] a = ((Vector) this).getValues();
            System.arraycopy(a, 0, out, 0, a.length);
        } else if (s.length == 2) {
            float[][] a = ((Matrix) this).getValues();
            for (int i = 0; i < a.length; i++) {
                System.arraycopy(a[i], 0, out, idx, a[i].length);
                idx += a[i].length;
            }
        } else if (s.length == 3) {
            float[][][] a = ((Tensor3D) this).getValues();
            for (int i = 0; i < a.length; i++)
                for (int j = 0; j < a[0].length; j++) {
                    System.arraycopy(a[i][j], 0, out, idx, a[i][j].length);
                    idx += a[i][j].length;
                }
        } else if (s.length == 4) {
            float[][][][] a = ((Tensor4D) this).getValues();
            for (int i = 0; i < a.length; i++)
                for (int j = 0; j < a[0].length; j++)
                    for (int k = 0; k < a[0][0].length; k++) {
                        System.arraycopy(a[i][j][k], 0, out, idx, a[i][j][k].length);
                        idx += a[i][j][k].length;
                    }
        }
        return new Vector(out);
    }
    
    /**
     * Alias for flatten (NumPy ravel returns a view; here we return a copy).
     */
    public Vector ravel() {
        return flatten();
    }
    
    /**
     * Returns the values of this tensor as a flattened float[] in row-major order.
     * The returned array is a new contiguous copy and does not share storage with the original tensor.
     */
    public float[] toFlatArray() {
        return flatten().getValues();
    }
    
    /**
     * General axis permutation. Subclasses implement for their rank; default not supported.
     */
    public Tensor permute(int... axes) {
        throw new UnsupportedOperationException("permute not supported for " + type());
    }
    
    /**
     * Return a new tensor with the given shape, reading elements in row-major (C) order
     * from this tensor's flattened contents. Behaves like numpy.reshape(..., order='C').
     * <p>
     * newShape length must be 0..4 and product(newShape) must equal this.size().
     */
    public Tensor reshape(int... newShape) {
        // Compute new total size
        long newSize = 1;
        for (int s : newShape) newSize *= s;
        
        if (newSize != this.size())
            throw new IllegalArgumentException(
                "Cannot reshape size " + this.size() + " into product " + newSize
            );
        
        float[] flat = this.toFlatArray();  // already row-major
        return createFromFlat(flat, newShape);
    }
    
    public static Tensor createFromFlat(float[] flat, int... shape) {
        int rank = shape.length;
        
        if (rank == 0) {
            if (flat.length != 1)
                throw new IllegalArgumentException("Scalar must have size 1");
            return new Scalar(flat[0]);
        }
        
        if (rank == 1) {
            return new Vector(flat.clone());
        }
        
        if (rank == 2) {
            int r = shape[0], c = shape[1];
            float[][] arr = new float[r][c];
            int idx = 0;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    arr[i][j] = flat[idx++];
            return new Matrix(arr);
        }
        
        if (rank == 3) {
            int d0 = shape[0], d1 = shape[1], d2 = shape[2];
            float[][][] arr = new float[d0][d1][d2];
            int idx = 0;
            for (int i = 0; i < d0; i++)
                for (int j = 0; j < d1; j++)
                    for (int k = 0; k < d2; k++)
                        arr[i][j][k] = flat[idx++];
            return new Tensor3D(arr);
        }
        
        if (rank == 4) {
            int d0 = shape[0], d1 = shape[1], d2 = shape[2], d3 = shape[3];
            float[][][][] arr = new float[d0][d1][d2][d3];
            int idx = 0;
            for (int i = 0; i < d0; i++)
                for (int j = 0; j < d1; j++)
                    for (int k = 0; k < d2; k++)
                        for (int l = 0; l < d3; l++)
                            arr[i][j][k][l] = flat[idx++];
            return new Tensor4D(arr);
        }
        
        throw new IllegalArgumentException("Unsupported rank " + rank);
    }
    
    public abstract float sum();
    
    // ---- Optional: minimal einsum for common cases ----
    public static Tensor einsum(String pattern, Tensor a, Tensor b) {
        // Very small subset: "ij,j->i" and "ij,jk->ik" and batched "bij,bjk->bik"
        pattern = pattern.replace(" ", "");
        if ("ij,j->i".equals(pattern) && a instanceof Matrix ma && b instanceof Vector vb) {
            return ma.matmul(vb);
        }
        if ("ij,jk->ik".equals(pattern) && a instanceof Matrix ma2 && b instanceof Matrix mb2) {
            return ma2.matmul(mb2);
        }
        if ("bij,bjk->bik".equals(pattern) && a instanceof Tensor3D ta && b instanceof Tensor3D tb) {
            return ta.matmul(tb);
        }
        if ("...ij,...jk->...ik".equals(pattern) && a instanceof Tensor4D ta4 && b instanceof Tensor4D tb4) {
            return ta4.matmul(tb4);
        }
        throw new UnsupportedOperationException("einsum pattern not supported: " + pattern);
    }
    
    /**
     * Calculates the element-wise reciprocal: 1 / (value + epsilon).
     * This prevents division by zero errors.
     */
    public Tensor reciprocal(float epsilon) {
        return this.applyOperation(val -> 1.0f / (val + epsilon));
    }
    
    /**
     * Convenience reciprocal with a standard small epsilon (1e-8).
     */
    public Tensor reciprocal() {
        return reciprocal(1e-8f);
    }
    
    @Override
    public String toString() {
        return type() + "(shape=" + java.util.Arrays.toString(shape()) + ")";
    }
}