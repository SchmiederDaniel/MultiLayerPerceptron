package de.darkandblue.neuralnetwork.math.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScalarTest {
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
}