package neuralnetwork.math.tensor;

import neuralnetwork.math.tensor.Matrix;
import neuralnetwork.math.tensor.Scalar;
import neuralnetwork.math.tensor.Tensor;
import neuralnetwork.math.tensor.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorTest {
    
    @Test
    void add() {
        // Testing for scalar
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Tensor b = new Scalar(3.5f);
        Tensor c = a.add(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 4.5f, 5.5f, 6.5f }, ((Vector) c).values);
        
        // Testing for vector
        a = new Vector(new float[] { 1f, 2f, 3f });
        b = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        c = a.add(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 3.5f, -0.5f, 5.5f }, ((Vector) c).values);
    }
    
    @Test
    void subtract() {
        // Testing for scalar
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Tensor b = new Scalar(-3.5f);
        Tensor c = a.add(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { -2.5f, -1.5f, -0.5f }, ((Vector) c).values);
        
        // Testing for vector
        a = new Vector(new float[] { 1f, 2f, 3f });
        b = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        c = a.subtract(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { -1.5f, 4.5f, 0.5f }, ((Vector) c).values);
    }
    
    @Test
    void mul() {
        // Testing for scalar
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Tensor b = new Scalar(-3.5f);
        Tensor c = a.mul(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { -3.5f, -7f, -10.5f }, ((Vector) c).values);
        
        // Testing for vector
        a = new Vector(new float[] { 1f, 2f, 3f });
        b = new Vector(new float[] { 2f, -3f, 4f });
        c = a.mul(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 2, -6, 12 }, ((Vector) c).values);
        
//        // Testing for Matrix
//        a = new Vector(new float[] { 2f, 6f });
//        b = new Matrix(new float[][] { { 2f, 6f }, { 3f, 1f } });
//        c = b.mul(a);
//        assertInstanceOf(Matrix.class, c);
//        assertArrayEquals(new float[][] { { 4, 36 }, { 6, 6 } }, ((Matrix) c).values);
    }
    
    @Test
    void divide() {
        // Testing for scalar
        Tensor a = new Vector(new float[] { 2f, 6f, -10f });
        Tensor b = new Scalar(-2f);
        Tensor c = a.divide(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { -1f, -3f, 5f }, ((Vector) c).values);
        
        // Testing for vector
        a = new Vector(new float[] { 2f, 6f, -8f });
        b = new Vector(new float[] { 2f, -3f, 4f });
        c = a.divide(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 1, -2, -2 }, ((Vector) c).values);
    }
    
    @Test
    void transpose() {
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Vector result = (Vector) a.transpose();
        assertEquals(a, result);
    }
    
    @Test
    void testEquals() {
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Vector b = new Vector(new float[] { 1f, 2f, 3f });
        Tensor c = a.deepCopy();
        a.add(new Vector(new float[] { 2.5f, 2.5f, 2.5f }));
        
        assertArrayEquals(a.values, b.values);
        assertEquals(a, c);
        assertNotEquals(new Vector(new float[] { 1f, 2f, 3f, 4f }), a);
    }
    
    @Test
    void deepCopy() {
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Vector b = (Vector) a.deepCopy();
        a.add(new Vector(new float[] { 2.5f, 2.5f, 2.5f }));
        
        assertArrayEquals(a.values, b.values);
        assertNotSame(a.values, b.values);
    }
}