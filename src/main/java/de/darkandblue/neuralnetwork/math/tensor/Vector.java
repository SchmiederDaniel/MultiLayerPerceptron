package de.darkandblue.neuralnetwork.math.tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

public class Vector implements Tensor {
  public final float[] values;
  
  public Vector(float[] values) {
    this.values = values;
  }
  
  @Override
  public Tensor add(Tensor tensor) {
    if (tensor instanceof Scalar)
      return this.add((Scalar) tensor);
    else
      throw new IllegalArgumentException(
          "The shape of Vector doesn't match " + tensor.type()
              + ". The other Tensor needs to be of lower dimension.");
  }
  
  public Vector add(Scalar scalar) {
    float[] newValues = new float[this.values.length];
    for (int i = 0; i < this.values.length; i++) {
      newValues[i] = this.values[i] + scalar.value;
    }
    return new Vector(newValues);
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
    float[] array = new float[this.values.length];
    Arrays.fill(array, value);
    return new Vector(array);
  }
  
  @Override
  public Tensor transpose() {
    return new Vector(Arrays.copyOf(this.values, this.values.length));
  }
  
  @Override
  public Tensor deepCopy() {
    return new Vector(Arrays.copyOf(this.values, this.values.length));
  }
  
  @Override
  public String type() {
    return "Vector";
  }
  
  @Override
  public String toString() {
    return type() + "(" + this.values.length + ")";
  }
  
  public static Vector full(float value, int size) {
    float[] values = new float[size];
    Arrays.fill(values, value);
    return new Vector(values);
  }
}