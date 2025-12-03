package neuralnetwork.math.tensor;

import neuralnetwork.math.tensor.Scalar;
import org.junit.jupiter.api.Test;

import static neuralnetwork.math.tensor.ScalarTest.Py.assertFloatEquals;

import static org.junit.jupiter.api.Assertions.*;

class ScalarTest {
    /** Small helpers for Python interop and comparisons */
    static class Py {
        static float toScalar(Object parsed) {
            if (parsed == null) throw new AssertionError("Parsed object is null");
            if (parsed instanceof Float f) return f;
            if (parsed instanceof float[] fa && fa.length == 1) return fa[0];
            if (parsed instanceof Object[] oa && oa.length == 1 && oa[0] instanceof float[] fa2 && fa2.length == 1) return fa2[0];
            throw new AssertionError("Unexpected parsed scalar type: " + parsed.getClass() + " -> " + parsed);
        }
        static void assertFloatEquals(float expected, float actual) {
            assertEquals(expected, actual, 1e-5f);
        }
    }
    @Test
    void transpose() {
        Scalar a = new Scalar(6f);
        Scalar result = (Scalar) a.transpose();
        assertEquals(6f, result.value);
    }
    
    @Test
    void deepCopy() {
        Scalar a = new Scalar(6f);
        Scalar b = (Scalar) a.deepCopy();
        a.add(new Scalar(2.5f));
        
        assertEquals(6f, a.value);
        assertNotSame(a.value, b.value);
    }
    
    @Test
    void testEquals() {
        Scalar a = new Scalar(6f);
        Scalar b = new Scalar(6f);
        Scalar c = new Scalar(-6f);
        
        assertEquals(a, b);
        assertNotEquals(a, c);
    }
    
    @Test
    void add() {
        Scalar a = new Scalar(1f);
        Scalar c = new Scalar(-3f);
        
        assertEquals(new Scalar(-2f), a.add(c));
    }
    
    @Test
    void subtract() {
        Scalar a = new Scalar(-3f);
        Scalar c = new Scalar(3f);
        
        assertEquals(new Scalar(-6f), a.subtract(c));
    }
    
    @Test
    void mul() {
        Scalar a = new Scalar(-3f);
        Scalar c = new Scalar(3f);
        
        assertEquals(new Scalar(-9f), a.mul(c));
    }
    
    @Test
    void matmul() {
        Scalar a = new Scalar(-3f);
        Scalar c = new Scalar(3f);
        
        assertEquals(new Scalar(-9f), a.matmul(c));
    }
    
    @Test
    void divide() {
        Scalar a = new Scalar(-3f);
        Scalar c = new Scalar(3f);
        
        assertEquals(new Scalar(-1f), a.divide(c));
    }
    
    @Test
    void python_scalar_ops_compare_with_java() throws Exception {
        // Prepare Java scalars
        Scalar a = new Scalar(-3f);
        Scalar b = new Scalar(3f);
        
        // +
        PythonBridge.PythonResult plus = new PythonBridge.PythonBuilder()
            .append("a = np.array(-3.0)")
            .append("b = np.array(3.0)")
            .append("c = a + b")
            .execute("c");
        float pyAdd = Py.toScalar(plus.parseToFloat());
        assertFloatEquals(((Scalar) a.add(b)).value, pyAdd);
        
        // -
        PythonBridge.PythonResult minus = new PythonBridge.PythonBuilder()
            .append("a = np.array(-3.0)")
            .append("b = np.array(3.0)")
            .append("c = a - b")
            .execute("c");
        float pySub = Py.toScalar(minus.parseToFloat());
        assertFloatEquals(((Scalar) a.subtract(b)).value, pySub);
        
        // * (elementwise for scalars)
        PythonBridge.PythonResult times = new PythonBridge.PythonBuilder()
            .append("a = np.array(-3.0)")
            .append("b = np.array(3.0)")
            .append("c = a * b")
            .execute("c");
        float pyMul = Py.toScalar(times.parseToFloat());
        assertFloatEquals(((Scalar) a.mul(b)).value, pyMul);
        
        // matmul in Java behaves like element-wise for Scalars, still product
        PythonBridge.PythonResult matmul = new PythonBridge.PythonBuilder()
            .append("a = np.array(-3.0)")
            .append("b = np.array(3.0)")
            .append("c = a * b") // numpy scalar @ scalar is not defined; use *
            .execute("c");
        float pyMatMul = Py.toScalar(matmul.parseToFloat());
        assertFloatEquals(((Scalar) a.matmul(b)).value, pyMatMul);
        
        // /
        PythonBridge.PythonResult div = new PythonBridge.PythonBuilder()
            .append("a = np.array(-3.0)")
            .append("b = np.array(3.0)")
            .append("c = a / b")
            .execute("c");
        float pyDiv = Py.toScalar(div.parseToFloat());
        assertFloatEquals(((Scalar) a.divide(b)).value, pyDiv);
    }
}