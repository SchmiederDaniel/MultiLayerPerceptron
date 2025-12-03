package neuralnetwork.math.tensor;

import java.util.Arrays;

public class Shape {
    public int[] shape;
    
    public Shape(int... shape) {
        this.shape = shape;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Shape) {
            return Arrays.equals(this.shape, ((Shape) obj).shape);
        }
        return false;
    }
    
    public static void main(String[] args) {
        Scalar scalar = Tensor.of(0.1);
    }
}