package de.darkandblue.neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Tensor4D implements Tensor {
  public final float[][][][] values;
  
  public Tensor4D(float[][][][] values) {
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
    float[][][][] values4d = new float[this.values.length][this.values[0].length][this.values[0][0].length][this.values[0][0][0].length];
    for (float[][][] values3d : values4d) {
      for (float[][] values2d : values3d) {
        for (float[] values1d : values2d) {
          Arrays.fill(values1d, value);
        }
      }
    }
    return new Tensor4D(values4d);
  }
  
  @Override
  public Tensor transpose() {
    return null;
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
    String dimensions = Arrays.stream(this.values)
        .map(tensor3d -> {
          String shape = Arrays.stream(tensor3d)
              .map(matrix -> "(" + matrix.length + (matrix.length > 0 ? ", " + matrix[0].length : "") + ")")
              .collect(Collectors.joining(", "));
          return "(" + shape + ")";
        })
        .collect(Collectors.joining());
    return type() + "(" + dimensions + ")";
  }
}