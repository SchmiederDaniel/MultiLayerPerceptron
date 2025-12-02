package de.darkandblue.neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Tensor3D implements Tensor {
  public final float[][][] values;
  
  public Tensor3D(float[][][] values) {
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
    float[][][] values3 = new float[this.values.length][this.values[0].length][this.values[0][0].length];
    for (float[][] values2d : this.values) {
      for (float[] values1d : values2d) {
        Arrays.fill(values1d, value);
      }
    }
    return new Tensor3D(values3);
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