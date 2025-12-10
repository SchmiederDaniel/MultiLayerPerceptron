package neuralnetwork.math;

public class Scalar extends Tensor {
    private final float value;
    
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
    public Tensor multiply(Tensor tensor) {
        return tensor.applyOperation(a -> this.value * a);
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        throw new IllegalArgumentException("Scalar does not support matmul (@) in NumPy semantics");
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
        return super.toString();
    }
    
    @Override
    public boolean equals(Object o) {
        if(o instanceof Scalar scalar) {
            return Float.compare(scalar.value, value) == 0;
        }
        return false;
    }

    public float get() { return value; }

    @Override
    public int[] shape() { return new int[0]; }
    
    @Override
    public Tensor permute(int... axes) {
        if (axes != null && axes.length != 0)
            throw new IllegalArgumentException("Scalar permute expects 0 axes");
        return deepCopy();
    }
    
    @Override
    public float sum() {
        return value;
    }
}