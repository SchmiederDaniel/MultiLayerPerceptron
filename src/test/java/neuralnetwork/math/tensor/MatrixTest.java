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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
    }
    
    @Test
    void mul() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor b = new Scalar(-2f);
        Tensor c = a.mul(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -2f, -4f, -6f },
            { -8f, -10f, -12f }
        }, ((Matrix) c).values);
        
        // Testing for vector (matrix-vector multiplication)
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Vector(new float[] { 2f, -3f, 4f });
        c = a.mul(b);
        assertInstanceOf(Vector.class, c);
        assertArrayEquals(new float[] { 8f, 17f }, ((Vector) c).values);
        
        // Testing for matrix (matrix-matrix multiplication)
        a = new Matrix(new float[][] {
            { 1f, 2f },
            { 3f, 4f }
        });
        b = new Matrix(new float[][] {
            { 2f, 0f },
            { 1f, 3f }
        });
        c = a.mul(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 4f, 6f },
            { 10f, 12f }
        }, ((Matrix) c).values);
    }
    
    @Test
    void matmul() {
        // Testing for scalar
        Matrix a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        Tensor b = new Scalar(-2f);
        Tensor c = a.matmul(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { -2f, -4f, -6f },
            { -8f, -10f, -12f }
        }, ((Matrix) c).values);
        
        // Testing for matrix (element-wise multiplication)
        a = new Matrix(new float[][] {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });
        b = new Matrix(new float[][] {
            { 2f, -1f, 3f },
            { -2f, 1f, -3f }
        });
        c = a.matmul(b);
        assertInstanceOf(Matrix.class, c);
        assertArrayEquals(new float[][] {
            { 2f, -2f, 9f },
            { -8f, 5f, -18f }
        }, ((Matrix) c).values);
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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
        
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
        }, ((Matrix) c).values);
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
        }, result.values);
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
        
        assertArrayEquals(a.values, b.values);
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
        
        assertArrayEquals(a.values, b.values);
        assertNotSame(a.values, b.values);
    }
    
    // Helper for setup to keep tests clean
    private static final String MAT_A_DEF = "a = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])";
    private static final String MAT_B_DEF = "b = np.array([[2.0, -1.0, 3.0],[-2.0, 1.0, -3.0]])";
    
    @Test
    void testAdd() throws Exception {
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        
        // 1. Matrix + Scalar
        Scalar s = new Scalar(3.5f);
        PythonBridge.PythonResult r1 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("c = a + 3.5")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.add(s)).values);
        
        // 2. Matrix + Vector (Broadcasting)
        Vector v = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        PythonBridge.PythonResult r2 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("v = np.array([2.5, -2.5, 2.5])")
            .append("c = a + v")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.add(v)).values);
        
        // 3. Matrix + Matrix (Element-wise)
        Matrix B = new Matrix(new float[][] { { 2f, -1f, 3f }, { -2f, 1f, -3f } });
        PythonBridge.PythonResult r3 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append(MAT_B_DEF)
            .append("c = a + b")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.add(B)).values);
    }
    
    @Test
    void testSubtract() throws Exception {
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        
        // 1. Matrix - Scalar
        Scalar s = new Scalar(-3.5f);
        PythonBridge.PythonResult r1 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("c = a - (-3.5)")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.subtract(s)).values);
        
        // 2. Matrix - Vector (Broadcasting)
        Vector v = new Vector(new float[] { 2.5f, -2.5f, 2.5f });
        PythonBridge.PythonResult r2 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("v = np.array([2.5, -2.5, 2.5])")
            .append("c = a - v")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.subtract(v)).values);
        
        // 3. Matrix - Matrix
        Matrix B = new Matrix(new float[][] { { 2f, -1f, 3f }, { -2f, 1f, -3f } });
        PythonBridge.PythonResult r3 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append(MAT_B_DEF)
            .append("c = a - b")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.subtract(B)).values);
    }
    
    @Test
    void testMul() throws Exception {
        /* * Based on your code, A.mul() acts as:
         * - Element-wise multiplication for Scalars
         * - Dot Product / Matrix Multiplication for Vectors and Matrices (@ operator)
         */
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        
        // 1. Matrix * Scalar (Element-wise)
        Scalar s = new Scalar(-2f);
        PythonBridge.PythonResult r1 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("c = a * -2.0") // Numpy * is element-wise
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.mul(s)).values);
        
        // 2. Matrix * Vector (Dot Product -> Result is Vector)
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonBridge.PythonResult r2 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a @ v") // Numpy @ is matmul/dot
            .append("print(c)")
            .execute();
        // Note: Result of Matrix (2x3) @ Vector (3) is Vector (2), so we use to1D
        assertArrayEquals((float[]) r2.parseToFloat(), ((Vector) A.mul(v)).values);
        
        // 3. Matrix * Matrix (Matrix Multiplication / Dot Product)
        Matrix M1 = new Matrix(new float[][] { { 1f, 2f }, { 3f, 4f } });
        Matrix M2 = new Matrix(new float[][] { { 2f, 0f }, { 1f, 3f } });
        PythonBridge.PythonResult r3 = new PythonBridge.PythonBuilder()
            .append("m1 = np.array([[1.0, 2.0],[3.0, 4.0]])")
            .append("m2 = np.array([[2.0, 0.0],[1.0, 3.0]])")
            .append("c = m1 @ m2") // Numpy @ operator
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) M1.mul(M2)).values);
    }
    
    @Test
    void testMatmul() throws Exception {
        /*
         * Based on your code, A.matmul() acts as:
         * - Element-wise multiplication (Hadamard product) for all inputs
         */
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        
        // 1. Matrix .matmul Scalar (Element-wise)
        Scalar s = new Scalar(-2f);
        PythonBridge.PythonResult r1 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("c = a * -2.0")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.matmul(s)).values);
        
        // 2. Matrix .matmul Matrix (Element-wise)
        Matrix B = new Matrix(new float[][] { { 2f, -1f, 3f }, { -2f, 1f, -3f } });
        PythonBridge.PythonResult r2 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append(MAT_B_DEF)
            .append("c = a * b") // Numpy * is element-wise
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) A.matmul(B)).values);
    }
    
    @Disabled("Skipping temporarily due to not being implemented yet in Matrix.java")
    @Test
    void testMatmulBroadcast() throws IOException, InterruptedException {
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        // 3. Edge Case: Matrix .matmul Vector (Broadcasted Element-wise)
        // This confirms 'matmul' is strictly element-wise in your implementation
        Vector v = new Vector(new float[] { 2.0f, -1.0f, 3.0f });
        PythonBridge.PythonResult r3 = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("v = np.array([2.0, -1.0, 3.0])")
            .append("c = a * v")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) A.matmul(v)).values);
    }
    
    @Test
    void testDivide() throws Exception {
        Matrix A = new Matrix(new float[][] { { 2f, 6f, -10f }, { 4f, 8f, -12f } });
        
        // 1. Matrix / Scalar
        Scalar s = new Scalar(-2f);
        PythonBridge.PythonResult r1 = new PythonBridge.PythonBuilder()
            .append("a = np.array([[2.0, 6.0, -10.0],[4.0, 8.0, -12.0]])")
            .append("c = a / -2.0")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.divide(s)).values);
        
        // 2. Matrix / Vector (Broadcasting)
        Matrix D2 = new Matrix(new float[][] { { 2f, 6f, -8f }, { 4f, 12f, 16f } });
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonBridge.PythonResult r2 = new PythonBridge.PythonBuilder()
            .append("a = np.array([[2.0, 6.0, -8.0],[4.0, 12.0, 16.0]])")
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a / v")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Matrix) D2.divide(v)).values);
        
        // 3. Matrix / Matrix (Element-wise)
        Matrix N = new Matrix(new float[][] { { 6f, 9f, -12f }, { 8f, 10f, 15f } });
        Matrix D = new Matrix(new float[][] { { 2f, -3f, 4f }, { 4f, 5f, -3f } });
        PythonBridge.PythonResult r3 = new PythonBridge.PythonBuilder()
            .append("n = np.array([[6.0, 9.0, -12.0],[8.0, 10.0, 15.0]])")
            .append("d = np.array([[2.0, -3.0, 4.0],[4.0, 5.0, -3.0]])")
            .append("c = n / d")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Matrix) N.divide(D)).values);
    }
    
    @Test
    void testTranspose() throws Exception {
        Matrix A = new Matrix(new float[][] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        
        PythonBridge.PythonResult r = new PythonBridge.PythonBuilder()
            .append(MAT_A_DEF)
            .append("c = a.T")
            .append("print(c)")
            .execute();
        assertArrayEquals((Object[]) r.parseToFloat(), ((Matrix) A.transpose()).values);
    }
}