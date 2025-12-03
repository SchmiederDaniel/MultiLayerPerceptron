package de.darkandblue.neuralnetwork.math.tensor;

import org.junit.jupiter.api.Test;

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
}