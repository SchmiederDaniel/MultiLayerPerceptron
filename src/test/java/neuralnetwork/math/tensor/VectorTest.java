package neuralnetwork.math.tensor;

import neuralnetwork.math.tensor.Matrix;
import neuralnetwork.math.tensor.Scalar;
import neuralnetwork.math.tensor.Tensor;
import neuralnetwork.math.tensor.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorTest {
    // Helpers
    private static void assertFloatArrayEquals(float[] expected, float[] actual, float eps) {
        assertEquals(expected.length, actual.length, "Length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], eps, "Mismatch at index " + i);
        }
    }
    private static float[] to1D(Object parsed) {
        if (parsed instanceof float[] fa) return fa;
        if (parsed instanceof Object[] oa && oa.length > 0 && oa[0] instanceof Float) {
            float[] out = new float[oa.length];
            for (int i = 0; i < oa.length; i++) out[i] = ((Float) oa[i]);
            return out;
        }
        throw new AssertionError("Unexpected parsed 1D type: " + (parsed == null ? "null" : parsed.getClass()));
    }
    
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
    
    @Test
    void python_vector_scalar_ops_compare_with_java() throws Exception {
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Scalar s = new Scalar(3.5f);
        // add
        PythonResult r1 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("c = a + 3.5")
            .execute("c");
        float[] py1 = to1D(r1.parseToFloat());
        assertFloatArrayEquals(((Vector) a.add(s)).values, py1, 1e-5f);
        
        // subtract
        PythonResult r2 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("c = a - 3.5")
            .execute("c");
        float[] py2 = to1D(r2.parseToFloat());
        assertFloatArrayEquals(((Vector) a.subtract(s)).values, py2, 1e-5f);
        
        // mul
        PythonResult r3 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("c = a * -3.5")
            .execute("c");
        float[] py3 = to1D(r3.parseToFloat());
        assertFloatArrayEquals(((Vector) a.mul(new Scalar(-3.5f))).values, py3, 1e-5f);
        
        // divide
        PythonResult r4 = new PythonBuilder()
            .append("a = np.array([2.0, 6.0, -10.0])")
            .append("c = a / -2.0")
            .execute("c");
        float[] py4 = to1D(r4.parseToFloat());
        Vector v = new Vector(new float[] { 2f, 6f, -10f });
        assertFloatArrayEquals(((Vector) v.divide(new Scalar(-2f))).values, py4, 1e-5f);
    }
    
    @Test
    void python_vector_vector_ops_compare_with_java() throws Exception {
        Vector a = new Vector(new float[] { 1f, 2f, 3f });
        Vector b = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        // add
        PythonResult r1 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("b = np.array([2.5, -2.5, 2.5])")
            .append("c = a + b")
            .execute("c");
        float[] py1 = to1D(r1.parseToFloat());
        assertFloatArrayEquals(((Vector) a.add(b)).values, py1, 1e-5f);
        
        // subtract
        PythonResult r2 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("b = np.array([2.5, -2.5, 2.5])")
            .append("c = a - b")
            .execute("c");
        float[] py2 = to1D(r2.parseToFloat());
        assertFloatArrayEquals(((Vector) a.subtract(b)).values, py2, 1e-5f);
        
        // elementwise mul
        PythonResult r3 = new PythonBuilder()
            .append("a = np.array([1.0, 2.0, 3.0])")
            .append("b = np.array([2.0, -3.0, 4.0])")
            .append("c = a * b")
            .execute("c");
        float[] py3 = to1D(r3.parseToFloat());
        Vector b2 = new Vector(new float[] { 2f, -3f, 4f });
        assertFloatArrayEquals(((Vector) a.mul(b2)).values, py3, 1e-5f);
        
        // divide elementwise
        PythonResult r4 = new PythonBuilder()
            .append("a = np.array([2.0, 6.0, -8.0])")
            .append("b = np.array([2.0, -3.0, 4.0])")
            .append("c = a / b")
            .execute("c");
        float[] py4 = to1D(r4.parseToFloat());
        Vector a2 = new Vector(new float[] { 2f, 6f, -8f });
        assertFloatArrayEquals(((Vector) a2.divide(b2)).values, py4, 1e-5f);
}
}