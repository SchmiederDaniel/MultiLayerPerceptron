package neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Matrix implements Tensor {
  public final float[][] values;
  
  public Matrix(float[][] values) {
    this.values = values;
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
  public Tensor multiply(Tensor tensor) {
    return null;
  }
  
  @Override
  public Tensor divide(Tensor tensor) {
    return null;
  }
  
  @Override
  public Tensor copyFill(float value) {
    float[][] values2d = new float[rows()][cols()];
    for (float[] values1d : this.values) {
      Arrays.fill(values1d, value);
    }
    return new Matrix(values2d);
  }
  
  @Override
  public Tensor transpose() {
    return null;
  }
  
  public int rows() {
    return this.values.length;
  }
  
  public int cols() {
    return this.values[0].length;
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
}