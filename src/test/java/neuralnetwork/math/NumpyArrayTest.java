package neuralnetwork.math;

import neuralnetwork.math.NumpyArray;
import neuralnetwork.math.NumpyArrayOld;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;


public class NumpyArrayTest {
  /**
   * Compatibility tests to ensure the new 3D NumpyArray implementation
   * behaves exactly like the old 2D implementation when Depth = 1.
   */
  
  private static final float DELTA = 0.0001f;
  private final Random random = new Random(1234); // Fixed seed for reproducibility
  
  
  // --- Helper Methods ---
  
  private NumpyArrayOld createRandomOld(int rows, int cols) {
    float[][] data = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = -10.0f + random.nextFloat() * 20.0f;
      }
    }
    return new NumpyArrayOld(data);
  }
  
  private NumpyArray createNewFromOld(NumpyArrayOld old) {
    float[][][] data = new float[1][old.rows()][old.cols()];
    for (int i = 0; i < old.rows(); i++) {
      System.arraycopy(old.data[i], 0, data[0][i], 0, old.cols());
    }
    return new NumpyArray(data);
  }
  
  private void assertArraysEquivalent(NumpyArrayOld expected, NumpyArray actual) {
    Assertions.assertEquals(1, actual.depth(), "Depth should be 1 for 2D compatibility");
    Assertions.assertEquals(expected.rows(), actual.rows(), "Rows mismatch");
    Assertions.assertEquals(expected.cols(), actual.cols(), "Cols mismatch");
    
    for (int i = 0; i < expected.rows(); i++) {
      for (int j = 0; j < expected.cols(); j++) {
        float oldVal = expected.data[i][j];
        float newVal = actual.data[0][i][j]; // Accessing depth 0
        Assertions.assertEquals(oldVal, newVal, DELTA,
            String.format("Mismatch at [%d][%d]. Old: %f, New: %f", i, j, oldVal, newVal));
      }
    }
  }
  
  // --- Tests ---
  
  @Test
  public void testDotProductMatrix() {
    // 2x3 dot 3x2 -> 2x2
    NumpyArrayOld aOld = createRandomOld(2, 3);
    NumpyArrayOld bOld = createRandomOld(3, 2);
    
    NumpyArray aNew = createNewFromOld(aOld);
    NumpyArray bNew = createNewFromOld(bOld);
    
    NumpyArrayOld resultOld = aOld.dot(bOld);
    NumpyArray resultNew = aNew.dot(bNew);
    
    assertArraysEquivalent(resultOld, resultNew);
  }
  
  @Test
  public void testDotProductVector() {
    // 2x2 dot 2x1 -> 2x1
    NumpyArrayOld matrixOld = createRandomOld(2, 2);
    NumpyArrayOld vectorOld = createRandomOld(2, 1);
    
    NumpyArray matrixNew = createNewFromOld(matrixOld);
    NumpyArray vectorNew = createNewFromOld(vectorOld);
    
    NumpyArrayOld resultOld = matrixOld.dot(vectorOld);
    NumpyArray resultNew = matrixNew.dot(vectorNew);
    
    assertArraysEquivalent(resultOld, resultNew);
  }
  
  @Test
  public void testAddMatrix() {
    NumpyArrayOld aOld = createRandomOld(4, 4);
    NumpyArrayOld bOld = createRandomOld(4, 4);
    
    NumpyArray aNew = createNewFromOld(aOld);
    NumpyArray bNew = createNewFromOld(bOld);
    
    assertArraysEquivalent(aOld.add(bOld), aNew.add(bNew));
  }
  
  @Test
  public void testAddScalarBroadcasting() {
    // Adding a 1x1 matrix to a 3x3 matrix
    NumpyArrayOld largeOld = createRandomOld(3, 3);
    NumpyArrayOld scalarOld = createRandomOld(1, 1);
    
    NumpyArray largeNew = createNewFromOld(largeOld);
    NumpyArray scalarNew = createNewFromOld(scalarOld);
    
    assertArraysEquivalent(largeOld.add(scalarOld), largeNew.add(scalarNew));
    // Test commutative property logic (Scalar + Large)
    assertArraysEquivalent(scalarOld.add(largeOld), scalarNew.add(largeNew));
  }
  
  @Test
  public void testAddVectorBroadcasting() {
    // Adding a 1x3 vector (bias) to a 3x3 matrix
    // Note: The old implementation specifically handled "add 1 dimensional array to a two-dimensional array"
    // if cols matched.
    
    NumpyArrayOld matrixOld = createRandomOld(3, 3);
    NumpyArrayOld rowVectorOld = createRandomOld(1, 3);
    
    NumpyArray matrixNew = createNewFromOld(matrixOld);
    NumpyArray rowVectorNew = createNewFromOld(rowVectorOld);
    
    assertArraysEquivalent(matrixOld.add(rowVectorOld), matrixNew.add(rowVectorNew));
  }
  
  @Test
  public void testSubtract() {
    NumpyArrayOld aOld = createRandomOld(2, 5);
    NumpyArrayOld bOld = createRandomOld(2, 5);
    
    NumpyArray aNew = createNewFromOld(aOld);
    NumpyArray bNew = createNewFromOld(bOld);
    
    assertArraysEquivalent(aOld.subtract(bOld), aNew.subtract(bNew));
  }
  
  @Test
  public void testMultiplyScalar() {
    NumpyArrayOld aOld = createRandomOld(3, 3);
    float scalar = 2.5f;
    
    NumpyArray aNew = createNewFromOld(aOld);
    
    assertArraysEquivalent(aOld.multiply(scalar), aNew.multiply(scalar));
  }
  
  @Test
  public void testMultiplyElementWise() {
    NumpyArrayOld aOld = createRandomOld(3, 3);
    NumpyArrayOld bOld = createRandomOld(3, 3);
    
    NumpyArray aNew = createNewFromOld(aOld);
    NumpyArray bNew = createNewFromOld(bOld);
    
    assertArraysEquivalent(aOld.multiply(bOld), aNew.multiply(bNew));
  }
  
  @Test
  public void testTranspose() {
    NumpyArrayOld aOld = createRandomOld(2, 4);
    NumpyArray aNew = createNewFromOld(aOld);
    
    assertArraysEquivalent(aOld.transpose(), aNew.transpose());
  }
  
  @Test
  public void testFlatten() {
    NumpyArrayOld aOld = createRandomOld(3, 3);
    NumpyArray aNew = createNewFromOld(aOld);
    
    NumpyArrayOld flatOld = aOld.flatten();
    NumpyArray flatNew = aNew.flatten();
    
    // Assert dimensions
    Assertions.assertEquals(flatOld.rows() * flatOld.cols(), flatNew.rows() * flatNew.cols());
    
    // Assert values match sequentially
    // Note: flatOld might be (N, 1) and flatNew might be (1, N, 1) depending on implementation details,
    // so we check the raw data stream.
    float[] oldArr = flatOld.flattenArray();
    float[] newArr = flatNew.flattenArray();
    
    Assertions.assertArrayEquals(oldArr, newArr, DELTA);
  }
  
  @Test
  public void testReshape() {
    NumpyArrayOld aOld = createRandomOld(4, 4); // 16 elements
    NumpyArray aNew = createNewFromOld(aOld);
    
    NumpyArrayOld reshapedOld = aOld.reshape(2, 8);
    
    // New reshape requires 3 args (depth, row, col), but since the old was 2D,
    // we reshape the new one into Depth 1, rows 2, cols 8.
    NumpyArray reshapedNew = aNew.reshape(1, 2, 8);
    
    assertArraysEquivalent(reshapedOld, reshapedNew);
  }
  
  @Test
  public void testAllDimensionPairs_AddSubtractMultiply() {
    int[] sizes = {1, 2, 3, 4};
    
    for (int r1 : sizes)
      for (int c1 : sizes)
        for (int r2 : sizes)
          for (int c2 : sizes) {
            
            NumpyArrayOld aOld = createRandomOld(r1, c1);
            NumpyArrayOld bOld = createRandomOld(r2, c2);
            
            NumpyArray aNew = createNewFromOld(aOld);
            NumpyArray bNew = createNewFromOld(bOld);
            
            // --- ADD ---
            try {
              assertArraysEquivalent(aOld.add(bOld), aNew.add(bNew));
            } catch (Exception ignored) {
              // expected when shapes are incompatible
            }
            
            // --- SUBTRACT ---
            try {
              assertArraysEquivalent(aOld.subtract(bOld), aNew.subtract(bNew));
            } catch (Exception ignored) {
              // expected when shapes are incompatible
            }
            
            // --- MULTIPLY ---
//            assertArraysEquivalent(aOld.multiply(bOld), aNew.multiply(bNew));
          }
  }

//  @Test
//  public void testAddColumnVectorBroadcasting() {
//    NumpyArrayOld matrixOld = createRandomOld(4, 5);
//    NumpyArrayOld colVectorOld = createRandomOld(4, 1);
//    
//    NumpyArray matrixNew = createNewFromOld(matrixOld);
//    NumpyArray colVectorNew = createNewFromOld(colVectorOld);
//    
//    assertArraysEquivalent(matrixOld.add(colVectorOld), matrixNew.add(colVectorNew));
//  }
  
  @Test
  public void testRowVectorBroadcasting_Rectangular() {
    NumpyArrayOld matrixOld = createRandomOld(5, 7);
    NumpyArrayOld rowVectorOld = createRandomOld(1, 7);
    
    NumpyArray matrixNew = createNewFromOld(matrixOld);
    NumpyArray rowVectorNew = createNewFromOld(rowVectorOld);
    
    assertArraysEquivalent(matrixOld.add(rowVectorOld), matrixNew.add(rowVectorNew));
  }

//  @Test
//  public void testColumnVectorBroadcasting_Rectangular() {
//    NumpyArrayOld matrixOld = createRandomOld(6, 4);
//    NumpyArrayOld colVectorOld = createRandomOld(6, 1);
//    
//    NumpyArray matrixNew = createNewFromOld(matrixOld);
//    NumpyArray colVectorNew = createNewFromOld(colVectorOld);
//    
//    assertArraysEquivalent(matrixOld.add(colVectorOld), matrixNew.add(colVectorNew));
//  }
  
  @Test
  public void testScalarBroadcasting_AllShapes() {
    int[] sizes = {1, 2, 3, 4};
    
    for (int r : sizes)
      for (int c : sizes) {
        NumpyArrayOld matrixOld = createRandomOld(r, c);
        NumpyArrayOld scalarOld = createRandomOld(1, 1);
        
        NumpyArray matrixNew = createNewFromOld(matrixOld);
        NumpyArray scalarNew = createNewFromOld(scalarOld);
        
        assertArraysEquivalent(matrixOld.add(scalarOld), matrixNew.add(scalarNew));
        assertArraysEquivalent(matrixOld.multiply(scalarOld), matrixNew.multiply(scalarNew));
        assertArraysEquivalent(matrixOld.subtract(scalarOld), matrixNew.subtract(scalarNew));
      }
  }
  
  @Test
  public void testDotVariousShapes() {
    int[][] dims = {
        {2, 3, 3, 2},
        {3, 4, 4, 1},
        {4, 2, 2, 5},
        {1, 5, 5, 3},
    };
    
    for (int[] d : dims) {
      int r1 = d[0], c1 = d[1];
      int r2 = d[2], c2 = d[3];
      
      NumpyArrayOld aOld = createRandomOld(r1, c1);
      NumpyArrayOld bOld = createRandomOld(r2, c2);
      
      if (c1 != r2) continue; // only valid dot products
      
      NumpyArray aNew = createNewFromOld(aOld);
      NumpyArray bNew = createNewFromOld(bOld);
      
      assertArraysEquivalent(aOld.dot(bOld), aNew.dot(bNew));
    }
  }
  
  @Test
  public void testMultiplyBroadcasting() {
    NumpyArrayOld matrixOld = createRandomOld(4, 4);
    NumpyArrayOld rowOld = createRandomOld(1, 4);
    NumpyArrayOld colOld = createRandomOld(4, 1);
    NumpyArrayOld scalarOld = createRandomOld(1, 1);
    
    NumpyArray matrixNew = createNewFromOld(matrixOld);
    NumpyArray rowNew = createNewFromOld(rowOld);
    NumpyArray colNew = createNewFromOld(colOld);
    NumpyArray scalarNew = createNewFromOld(scalarOld);

//    assertArraysEquivalent(matrixOld.multiply(rowOld), matrixNew.multiply(rowNew));
    assertArraysEquivalent(matrixOld.multiply(colOld), matrixNew.multiply(colNew));
    assertArraysEquivalent(matrixOld.multiply(scalarOld), matrixNew.multiply(scalarNew));
  }
  
}