package neuralnetwork.math;

import java.util.Arrays;

public class Vector extends Tensor {
    private final float[] values;
    
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
        if (ENABLE_SHAPE_CHECKS && this.values.length != vector.values.length)
            throw new IllegalArgumentException("Element-wise op requires same length: " + this.values.length + " vs " + vector.values.length);
        float[] result = new float[this.values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = operation.applyAsFloat(this.values[i], vector.values[i]);
        }
        return new Vector(result);
    }
    
    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a + ((Scalar) tensor).get());
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a + b);
        } else {
            return tensor.add(this);
        }
    }

    @Override
    public Tensor subtract(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a - ((Scalar) tensor).get());
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a - b);
        } else {
            return tensor.subtract(this);
        }
    }

    @Override
    public Tensor multiply(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a * ((Scalar) tensor).get());
        } else if (tensor instanceof Vector) {
            return applyVectorOperation((Vector) tensor, (a, b) -> a * b);
        } else {
            return tensor.multiply(this);
        }
    }
    
    @Override
    public Tensor matmul(Tensor tensor) {
        if (tensor instanceof Vector v) {
            if (ENABLE_SHAPE_CHECKS && this.values.length != v.values.length)
                throw new IllegalArgumentException("Vector@Vector length mismatch: " + this.values.length + " vs " + v.values.length);
            float sum = 0f;
            for (int i = 0; i < values.length; i++) sum += this.values[i] * v.values[i];
            return new Scalar(sum);
        } else if (tensor instanceof Matrix m) {
            // 1D (n) @ 2D (n,m) -> 1D (m)
            float[][] mv = m.getValues();
            if (ENABLE_SHAPE_CHECKS && this.values.length != mv.length)
                throw new IllegalArgumentException("Vector@Matrix mismatch: " + this.values.length + " vs rows=" + mv.length);
            int n = mv[0].length;
            float[] out = new float[n];
            for (int j = 0; j < n; j++) {
                float s = 0f;
                for (int i = 0; i < this.values.length; i++) s += this.values[i] * mv[i][j];
                out[j] = s;
            }
            return new Vector(out);
        } else if (tensor instanceof Scalar) {
            throw new IllegalArgumentException("Vector @ Scalar is not defined in NumPy");
        }
        throw new IllegalArgumentException("Unsupported matmul for Vector with " + tensor.type());
    }

    @Override
    public Tensor divide(Tensor tensor) {
        if (tensor instanceof Scalar) {
            return applyOperation(a -> a / ((Scalar) tensor).get());
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
    public String toString() { return super.toString(); }
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Vector) {
            return Arrays.equals(this.values, ((Vector) obj).values);
        }
        return false;
    }

    public float[] getValues() { return values; }
    public float get(int i) { return values[i]; }

    @Override
    public int[] shape() { return new int[] { values.length }; }
    
    @Override
    public float sum() {
        float sum = 0f;
        for (float v : values) sum += v;
        return sum;
    }
}