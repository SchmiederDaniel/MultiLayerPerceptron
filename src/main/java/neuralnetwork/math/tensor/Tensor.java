package neuralnetwork.math.tensor;

public interface Tensor {
  Tensor add(Tensor tensor);
  
  Tensor subtract(Tensor tensor);
  
  Tensor multiply(Tensor tensor);
  
  Tensor divide(Tensor tensor);
  
  Tensor copyFill(float value);
  
  Tensor transpose();
  
  String toString();
  
  /**
   * Copies a tensor object without reference.
   *
   * @return
   */
  Tensor deepCopy();
  
  String type();
  
  public static Scalar of(double value) {
    return new Scalar((float) value);
  }
  
  public static float valueOf(double f) {
    return (float) f;
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
}