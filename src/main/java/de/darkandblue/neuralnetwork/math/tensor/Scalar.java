package de.darkandblue.neuralnetwork.math.tensor;

public class Scalar extends Number implements Tensor {
  public final float value;
  
  public Scalar(float value) {
    this.value = value;
  }
  
  @Override
  public Tensor add(Tensor tensor) {
    if (tensor instanceof Scalar) {
      return this.add((Scalar) tensor);
    } else {
      // The scalar is being added to other tensors.
      return tensor.add(this);
    }
  }
  
  public Scalar add(Scalar scalar) {
    return new Scalar(this.value + scalar.value);
  }
  
  @Override
  public Tensor subtract(Tensor tensor) {
    if (tensor instanceof Scalar) {
      return this.subtract((Scalar) tensor);
    } else {
      throw new RuntimeException("This is not tested yet. Are really all the cases right here? 1d - 4d?");
//            return tensor.copyFill(this.value).subtract(tensor);
    }
  }
  
  public Scalar subtract(Scalar scalar) {
    return new Scalar(this.value - scalar.value);
  }
  
  @Override
  public Tensor multiply(Tensor tensor) {
    if (tensor instanceof Scalar)
      return this.multiply((Scalar) tensor);
    else
      throw new IllegalArgumentException(
          "The shape of Scalar doesn't match " + tensor.type()
              + ". The other Tensor needs to be of lower dimension.");
  }
  
  public Scalar multiply(Scalar scalar) {
    return new Scalar(this.value * scalar.value);
  }
  
  @Override
  public Tensor divide(Tensor tensor) {
    if (tensor instanceof Scalar)
      return this.divide((Scalar) tensor);
    else
      throw new IllegalArgumentException(
          "The shape of Scalar doesn't match " + tensor.type()
              + ". The other Tensor needs to be of lower dimension.");
  }
  
  @Override
  public Tensor copyFill(float value) {
    return new Scalar(value);
  }
  
  public Scalar divide(Scalar scalar) {
    return new Scalar(this.value / scalar.value);
  }
  
  @Override
  public Tensor transpose() {
    return new Scalar(this.value);
  }
  
  @Override
  public Tensor deepCopy() {
    return new Scalar(this.value);
  }
  
  @Override
  public String type() {
    return "Scalar";
  }
  
  @Override
  public int intValue() {
    return (int) this.value;
  }
  
  @Override
  public long longValue() {
    return (long) this.value;
  }
  
  @Override
  public float floatValue() {
    return this.value;
  }
  
  @Override
  public double doubleValue() {
    return this.value;
  }
  
  @Override
  public String toString() {
    return type();
  }
}