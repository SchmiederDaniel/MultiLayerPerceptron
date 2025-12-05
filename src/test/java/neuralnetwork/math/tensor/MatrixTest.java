package neuralnetwork.math.tensor;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {
    
    @Test
    void add() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor b = new Scalar(3.5f);
        Tensor c = a.add(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 4.5f, 5.5f, 6.5f },
            { 7.5f, 8.5f, 9.5f }
        }, ((Matrix) c).getValues());
        
        // Testing for vector (broadcasting)
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        c = a.add(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 3.5f, -0.5f, 5.5f },
            { 6.5f, 2.5f, 8.5f }
        }, ((Matrix) c).getValues());
        
        // Testing for matrix
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Matrix(new float[][] {
            { 2f, -1f, 3f },
            { -2f, 1f, -3f }
        });
        c = a.add(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 3f, 1f, 6f },
            { 2f, 6f, 3f }
        }, ((Matrix) c).getValues());
    }
    
    @Test
    void subtract() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor b = new Scalar(-3.5f);
        Tensor c = a.subtract(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 4.5f, 5.5f, 6.5f },
            { 7.5f, 8.5f, 9.5f }
        }, ((Matrix) c).getValues());
        
        // Testing for vector (broadcasting)
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        c = a.subtract(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -1.5f, 4.5f, 0.5f },
            { 1.5f, 7.5f, 3.5f }
        }, ((Matrix) c).getValues());
        
        // Testing for matrix
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Matrix(new float[][] {
            { 2f, -1f, 3f },
            { -2f, 1f, -3f }
        });
        c = a.subtract(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -1f, 3f, 0f },
            { 6f, 4f, 9f }
        }, ((Matrix) c).getValues());
    }
    
    @Test
    void mul() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor b = new Scalar(-2f);
        Tensor c = a.multiply(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -2f, -4f, -6f },
            { -8f, -10f, -12f }
        }, ((Matrix) c).getValues());
        
        // Testing for vector (matrix-vector multiplication)
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Vector(new float[] { 2f, -3f, 4f });
        c = a.matmul(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 8f, 17f }, ((Vector) c).getValues());
        
        // Testing for matrix (matrix-matrix multiplication)
        a = new Matrix(new float[][] {
            { 1f, 2f },
            { 3f, 4f }
        });
        b = new Matrix(new float[][] {
            { 2f, 0f },
            { 1f, 3f }
        });
        c = a.matmul(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 4f, 6f },
            { 10f, 12f }
        }, ((Matrix) c).getValues());
    }
    
    @Test
    void matmul() {
        // Now matmul is linear algebra
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        Tensor c = a.matmul(v);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 8f, 17f }, ((Vector) c).getValues());
        Matrix m2 = new Matrix(new float[][] {
            { 2f, 0f },
            { 1f, 3f },
            { -1f, 2f }
        });
        Tensor c2 = a.matmul(m2);
        assertInstanceOf(Matrix.class, c2);
        assertArrayEquals(new float[][] { { 1f, 12f }, { 7f, 27f } }, ((Matrix) c2).getValues());
    }
    
    @Test
    void divide() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 2f, 6f, -10f },
            { 4f, 8f, -12f }
        });
        Tensor b = new Scalar(-2f);
        Tensor c = a.divide(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -1f, -3f, 5f },
            { -2f, -4f, 6f }
        }, ((Matrix) c).getValues());
        
        // Testing for vector (broadcasting)
        a = new Matrix(new float[][] {
            { 2f, 6f, -8f },
            { 4f, 12f, 16f }
        });
        b = new Vector(new float[] { 2f, -3f, 4f });
        c = a.divide(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 1f, -2f, -2f },
            { 2f, -4f, 4f }
        }, ((Matrix) c).getValues());
        
        // Testing for matrix
        a = new Matrix(new float[][] {
            { 6f, 9f, -12f },
            { 8f, 10f, 15f }
        });
        b = new Matrix(new float[][] {
            { 2f, -3f, 4f },
            { 4f, 5f, -3f }
        });
        c = a.divide(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 3f, -3f, -3f },
            { 2f, 2f, -5f }
        }, ((Matrix) c).getValues());
    }
    
    @Test
    void transpose() {
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Matrix result = (Matrix) a.transpose();
        assertArrayEquals(new float[][] {
            { 1f, 4f },
            { 2f, 5f },
            { 3f, 6f }
        }, result.getValues());
    }
    
    @Test
    void testEquals() {
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Matrix b = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor c = a.deepCopy();
        a.add(new Matrix(new float[][] {
            { 2.5f, 2.5f, 2.5f },
            { 2.5f, 2.5f, 2.5f }
        }));
        
        assertArrayEquals(a.getValues(), b.getValues());
        assertEquals(a, c);
        assertNotEquals(new Matrix(new float[][] {
            { 1f, 2f },
            { 3f, 4f }
        }), a);
    }
    
    @Test
    void deepCopy() {
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Matrix b = (Matrix) a.deepCopy();
        a.add(new Matrix(new float[][] {
            { 2.5f, 2.5f, 2.5f },
            { 2.5f, 2.5f, 2.5f }
        }));
        
        assertArrayEquals(a.getValues(), b.getValues());
        assertNotSame(a.getValues(), b.getValues());
    }
    
    // Helper for setup to keep tests clean
    private static final String A_PY = "a = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])";
    private static final String B_PY = "b = np.array([[2.0, -1.0, 3.0],[-2.0, 1.0, -3.0]])";
    private static final Matrix B = new Matrix(new float[][] { { 2f, -1f, 3f }, { -2f, 1f, -3f } });
    
    @Test
    void testAdd() throws Exception {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        // 1. Matrix + Scalar
        Scalar s = new Scalar(3.5f);
        PythonResult r1 = new PythonBuilder()
            .append(A_PY)
            .append("c = a + 3.5")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.add(s)).getValues());
        
        // 2. Matrix + Vector (Broadcasting)
        Vector v = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        PythonResult r2 = new PythonBuilder()
            .append(A_PY)
            .append("v = np.array([2.5, -2.5, 2.5])")
            .append("c = a + v")
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.add(v)).getValues());
        
        // 3. Matrix + Matrix (Element-wise)
        PythonResult r3 = new PythonBuilder()
            .append(A_PY)
            .append(B_PY)
            .append("c = a + b")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.add(B)).getValues());
    }
    
    @Test
    void testSubtract() throws Exception {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        // 1. Matrix - Scalar
        Scalar s = new Scalar(-3.5f);
        PythonResult r1 = new PythonBuilder()
            .append(A_PY)
            .append("c = a - (-3.5)")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.subtract(s)).getValues());
        
        // 2. Matrix - Vector (Broadcasting)
        Vector v = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        PythonResult r2 = new PythonBuilder()
            .append(A_PY)
            .append("v = np.array([2.5, -2.5, 2.5])")
            .append("c = a - v")
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.subtract(v)).getValues());
        
        // 3. Matrix - Matrix
        PythonResult r3 = new PythonBuilder()
            .append(A_PY)
            .append(B_PY)
            .append("c = a - b")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.subtract(B)).getValues());
    }
    
    @Test
    void testMul() throws Exception {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        /* * Based on your code, A.mul() acts as:
         * - Element-wise multiplication for Scalars
         * - Dot Product / Matrix Multiplication for Vectors and Matrices (@ operator)
         */
        
        // 1. Matrix * Scalar (Element-wise)
        Scalar s = new Scalar(-2f);
        PythonResult r1 = new PythonBuilder()
            .append(A_PY)
            .append("c = a * -2.0") // Numpy * is element-wise
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.multiply(s)).getValues());
        
        // 2. Matrix * Vector (Dot Product -> Result is Vector)
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult r2 = new PythonBuilder()
            .append(A_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a @ v") // Numpy @ is matmul/dot
            .execute("c");
        // Note: Result of Matrix (2x3) @ Vector (3) is Vector (2), so we use to1D
        assertArrayEquals((float[]) r2.parseToFloat(), ((Vector) A.matmul(v)).getValues());
        
        // 3. Matrix * Matrix (Matrix Multiplication / Dot Product)
        Matrix M1 = new Matrix(new float[][] { { 1f, 2f }, { 3f, 4f } });
        Matrix M2 = new Matrix(new float[][] { { 2f, 0f }, { 1f, 3f } });
        PythonResult r3 = new PythonBuilder()
            .append("m1 = np.array([[1.0, 2.0],[3.0, 4.0]])")
            .append("m2 = np.array([[2.0, 0.0],[1.0, 3.0]])")
            .append("c = m1 @ m2") // Numpy @ operator
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) M1.matmul(M2)).getValues());
    }
    
    @Test
    void testMatmul() throws Exception {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        /*
         * Based on your code, A.matmul() acts as:
         * - Element-wise multiplication (Hadamard product) for all inputs
         */
        
        // 1. Matrix .matmul Scalar (Element-wise)
        Scalar s = new Scalar(-2f);
        PythonResult r1 = new PythonBuilder()
            .append(A_PY)
            .append("c = a * -2.0")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.multiply(s)).getValues());
        
        // 2. Matrix .matmul Matrix (Element-wise)
        PythonResult r2 = new PythonBuilder()
            .append(A_PY)
            .append(B_PY)
            .append("c = a * b") // Numpy * is element-wise
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.multiply(B)).getValues());
    }
    
    @Disabled("Skipping temporarily due to not being implemented yet in Matrix.java")
    @Test
    void testMatmulBroadcast() throws IOException, InterruptedException {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        // 3. Edge Case: Matrix .matmul Vector (Broadcasted Element-wise)
        // This confirms 'matmul' is strictly element-wise in your implementation
        Vector v = new Vector(new float[] { 2.0f, -1.0f, 3.0f });
        PythonResult r3 = new PythonBuilder()
            .append(A_PY)
            .append("v = np.array([2.0, -1.0, 3.0])")
            .append("c = a * v")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.multiply(v)).getValues());
    }
    
    @Test
    void testDivide() throws Exception {
        final Matrix E = new Matrix(new float[][] { { 2f, 6f, -10f }, { 4f, 8f, -12f } });
        
        // 1. Matrix / Scalar
        Scalar s = new Scalar(-2f);
        PythonResult r1 = new PythonBuilder()
            .append("e = np.array([[2.0, 6.0, -10.0],[4.0, 8.0, -12.0]])")
            .append("c = e / -2.0")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) E.divide(s)).getValues());
        
        // 2. Matrix / Vector (Broadcasting)
        Matrix D2 = new Matrix(new float[][] { { 2f, 6f, -8f }, { 4f, 12f, 16f } });
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult r2 = new PythonBuilder()
            .append("a = np.array([[2.0, 6.0, -8.0],[4.0, 12.0, 16.0]])")
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a / v")
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) D2.divide(v)).getValues());
        
        // 3. Matrix / Matrix (Element-wise)
        Matrix N = new Matrix(new float[][] { { 6f, 9f, -12f }, { 8f, 10f, 15f } });
        Matrix D = new Matrix(new float[][] { { 2f, -3f, 4f }, { 4f, 5f, -3f } });
        PythonResult r3 = new PythonBuilder()
            .append("n = np.array([[6.0, 9.0, -12.0],[8.0, 10.0, 15.0]])")
            .append("d = np.array([[2.0, -3.0, 4.0],[4.0, 5.0, -3.0]])")
            .append("c = n / d")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) N.divide(D)).getValues());
    }
    
    @Test
    void testTranspose() throws Exception {
        final Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        PythonResult r = new PythonBuilder()
            .append(A_PY)
            .append("c = a.T")
            .execute("c");
        assertArrayEquals((Object[]) r.parseToFloat(), ((Matrix) A.transpose()).getValues());
    }
}