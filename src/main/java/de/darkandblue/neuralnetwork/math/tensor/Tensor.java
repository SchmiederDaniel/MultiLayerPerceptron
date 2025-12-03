package de.darkandblue.neuralnetwork.math.tensor;


import jep.*;

public abstract class Tensor {
    abstract Tensor applyOperation(FloatOperator operation);
    
    abstract Tensor add(Tensor tensor);
    
    abstract Tensor subtract(Tensor tensor);
    
    /**
     * Applies a Hadamard product between two tensors of the same shape.
     *
     * <p>Numpy Example:</p>
     * <pre>{@code
     * a = np.array([[1, 2, 3], [1, 2, 3]])
     * b = np.array([3, 2, 4])
     * a * b = [3 4 12]
     * a.__mul__(b) = [3 4 12]
     * }</pre>
     *
     * <p>Example:</p>
     * <pre>{@code
     * // Two tensors with matching dimensions (both 2x3)
     * Tensor a = Tensor.of(new float[][]{
     *     {1, 2, 3},
     *     {1, 2, 3}
     * });
     * Tensor b = Tensor.of(new float[][]{
     *     {3, 2, 4}
     * });
     * // Valid: shapes match (2x3 ⊙ 1x3)
     * Tensor c = a.matmul(b);
     * }</pre>
     *
     * @param tensor
     * @return The result of the Hadamard multiplication.
     */
    abstract Tensor mul(Tensor tensor);
    
    /**
     * Applies matrix multiplication with two tensors.
     *
     * <p>Dimension requirement: both tensors must have identical shapes.
     * For example, a 2x3 tensor can only be multiplied element-wise with
     * another 2x3 tensor.</p>
     * 
     * <p>Numpy Example:</p>
     * <pre>{@code
     * a = np.array([1, 2, 3])
     * b = np.array([[3], [2], [4]])
     *
     * a @ b = [[3, 2, 4], [6, 4, 8], [9, 6, 12]]
     * a.__matmul__(b) = [[3, 2, 4], [6, 4, 8], [9, 6, 12]]
     * }</pre>
     * @param tensor The other tensor, which must have the same shape as this tensor
     * @return The element-wise (Hadamard) product
     */
    abstract Tensor matmul(Tensor tensor);
    
    abstract Tensor divide(Tensor tensor);
    
    /**
     * Creates a tensor of the same shape and fills it with a given value.
     *
     * @param value
     * @return
     */
    public Tensor copyFill(float value) {
        return this.applyOperation(_ -> value);
    }
    
    abstract Tensor transpose();
    
    /**
     * Copies a tensor object without reference.
     *
     * @return
     */
    abstract Tensor deepCopy();
    
    abstract String type();
    
    public static Scalar of(double value) {
        return new Scalar((float) value);
    }
    
    public static Vector of(double... values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++)
            result[i] = (float) values[i];
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
    
    public static void main(String[] args) {
        MainInterpreter.setJepLibraryPath("C:\\Users\\darka\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\jep\\jep.dll");
        try(SharedInterpreter interp = new SharedInterpreter()) {
            float[] f = new float[] { 1.0f, 2.1f, 3.3f, 4.5f, 5.6f, 6.7f };
            NDArray<float[]> nd = new NDArray<>(f, 3, 2);
            interp.set("x", nd);
        }
    }
}