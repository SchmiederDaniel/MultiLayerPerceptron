package de.darkandblue.neuralnetwork.math.tensor;

import java.util.Arrays;

public class Vector extends Tensor {
    public final float[] values;
    
    public Vector(float[] values) {
        this.values = values;
    }
    
    @Override
    public Tensor applyOperation(FloatOperator operation) {
        float[] result = new float[this.values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = operation.applyAsFloat(this.values[i]);
        }
        return new Vector(result);
    }
    
    public Vector applyVectorOperation(Vector vector, FloatBinaryOperator operation) {
        float[] result = new float[this.values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = operation.applyAsFloat(this.values[i], vector.values[i]);
        }
        return new Vector(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a + ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a + b);
        } else {
            return tensor.add(this);
        }
    }

    @Override
    public Tensor subtract(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a - ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a - b);
        } else {
            return tensor.subtract(this);
        }
    }

    @Override
    public Tensor mul(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a * ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a * b);
        } else {
            return tensor.mul(this);
        }
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        return applyVectorOperation((Vector) tensor, (a, b) -> a * b);
    }
    
    @Override
    public Tensor divide(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a / ((Scalar) tensor).value);
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a / b);
        } else {
            return tensor.divide(this);
        }
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
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Vector) {
            return Arrays.equals(this.values, ((Vector) obj).values);
        }
        return false;
    }
}