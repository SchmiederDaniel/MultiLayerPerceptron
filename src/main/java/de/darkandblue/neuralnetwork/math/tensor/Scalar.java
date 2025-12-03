package de.darkandblue.neuralnetwork.math.tensor;

public class Scalar extends Tensor {
    public final float value;
    
    public Scalar(float value) {
        this.value = value;
    }
    
    @Override
    public Tensor applyOperation(FloatOperator operation) {
        return new Scalar(operation.applyAsFloat(this.value));
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        return tensor.applyOperation(a -> this.value + a);
    }
    
    @Override
    public Tensor subtract(Tensor tensor) {
        return tensor.applyOperation(a -> this.value - a);
    }
    
    @Override
    public Tensor mul(Tensor tensor) {
        return tensor.applyOperation(a -> this.value * a);
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        return this.mul(tensor);
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        return tensor.applyOperation(a -> this.value / a);
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
    public String toString() {
        return type();
    }
    
    @Override
    public boolean equals(Object o) {
        if(o instanceof Scalar scalar) {
            return Float.compare(scalar.value, value) == 0;
        }
        return false;
    }
}