package neuralnetwork.math;

import neuralnetwork.initialization.WeightInitializer;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class NumpyArray {
  // 3D Array: [Depth][Rows][Cols]
  final public float[][][] data;
  
  public NumpyArray(float[][][] data) {
    this.data = data;
  }
  
  public NumpyArray(float[][] data) {
    this(new float[][][]{data});
  }
  
  public NumpyArray(float[] data) {
    this(new float[][][]{{data}});
  }
  
  public NumpyArray(float data) {
    this(new float[][][]{{{data}}});
  }
  
  public NumpyArray(int depth, int rows, int cols) {
    this.data = new float[depth][rows][cols];
  }
  
  public NumpyArray(int rows, int cols) {
    this.data = new float[1][rows][cols];
  }
  
  public int depth() {
    return data.length;
  }
  
  public int rows() {
    return data[0].length;
  }
  
  public int cols() {
    return data[0][0].length;
  }
  
  // Creates an identity matrix of 1 depth: (1, n, n)
  public static NumpyArray identity(int n) {
    float[][][] newData = new float[1][n][n];
    for (int i = 0; i < n; i++) {
      newData[0][i][i] = 1;
    }
    return new NumpyArray(newData);
  }
  
  public void checkMatchDimensions(NumpyArray other) {
    if (depth() != other.depth() || rows() != other.rows() || cols() != other.cols()) {
      throw new IllegalArgumentException("Dimensions don't match " + dimension() + ", " + other.dimension());
    }
  }
  
  /*
   * Batch Matrix Multiplication
   * Result logic: (Depth, Rows, Cols) * (Depth, Cols, OtherCols) = (Depth, Rows, OtherCols)
   */
  public NumpyArray dot(NumpyArray other) {
    NumpyArray a = this;
    NumpyArray b = other;
    
    // 1. Scalar * Matrix (1x1x1 vs DxRxC)
    if (a.depth() == 1 && a.rows() == 1 && a.cols() == 1) {
      // Swap if necessary so 'a' is the scalar
      if (other.depth() > 1 || other.rows() > 1) {
        a = this;
        b = other;
      } else {
        a = other;
        b = this;
      }
      // Standard scalar mult
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      float val = a.data[0][0][0];
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            newData[d][r][c] = val * b.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 2. Standard Batch Matrix Multiplication
    // Depth must usually match, or one must be 1 (broadcast). 
    // For simplicity: Assuming strict depth match for dot product here.
    if (a.cols() != b.rows()) {
    }
    
    int commonDepth = Math.max(a.depth(), b.depth());
    float[][][] newData = new float[commonDepth][a.rows()][b.cols()];
    
    for (int d = 0; d < commonDepth; d++) {
      // Handle broadcasting if one array has depth 1
      int dIndexA = (a.depth() == 1) ? 0 : d;
      int dIndexB = (b.depth() == 1) ? 0 : d;
      
      for (int row = 0; row < a.rows(); row++) {
        for (int otherCol = 0; otherCol < b.cols(); otherCol++) {
          float sum = 0;
          for (int col = 0; col < a.cols(); col++) {
            sum += a.data[dIndexA][row][col] * b.data[dIndexB][col][otherCol];
          }
          newData[d][row][otherCol] = sum;
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  // Creates a (1, N, 1) column vector from inputs
  public static NumpyArray of(float... input) {
    float[][][] newData = new float[1][input.length][1];
    for (int index = 0; index < input.length; index++) {
      newData[0][index][0] = input[index];
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray add(NumpyArray other) {
    // Ensure 'a' is the smaller/broadcastable one if dimensions differ significantly
    NumpyArray a = (size() < other.size()) ? this : other;
    NumpyArray b = (size() < other.size()) ? other : this;
    
    // 1. Scalar Addition (1x1x1)
    if (a.depth() == 1 && a.rows() == 1 && a.cols() == 1) {
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      float val = a.data[0][0][0];
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            newData[d][r][c] = val + b.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 2b. Broadcast row-vector (1 x 1 x C) over (1 x R x C)
    // Matches old 2D rule: (1 x C) added to (R x C)
    if (a.depth() == 1 && a.rows() == 1 && a.cols() == b.cols()) {
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            newData[d][r][c] = a.data[0][0][c] + b.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 2. Broadcast Matrix to Batch (1xRxC) + (DxRxC)
    // e.g. Adding bias weights to a batch of inputs
    if (a.depth() == 1 && a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            newData[d][r][c] = a.data[0][r][c] + b.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 3. Element-wise (Dimensions must match)
    if (a.depth() == b.depth() && a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][][] newData = new float[rows()][rows()][cols()]; // note: depth logic implicit
      newData = new float[depth()][rows()][cols()];
      
      for (int d = 0; d < depth(); d++) {
        for (int r = 0; r < rows(); r++) {
          for (int c = 0; c < cols(); c++) {
            newData[d][r][c] = a.data[d][r][c] + b.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    throw new RuntimeException("Couldn't find an add( function for dimensions " + dimension() + " / " + other.dimension());
  }
  
  public String dimension() {
    return "(" + depth() + ", " + rows() + ", " + cols() + ")";
  }
  
  public int size() {
    return depth() * rows() * cols();
  }
  
  public NumpyArray subtract(NumpyArray other) {
    boolean reversed = !(this.size() < other.size());
    NumpyArray a = this.size() < other.size() ? this : other;
    NumpyArray b = this.size() < other.size() ? other : this;
    
    // 1. Scalar Subtraction
    if (a.depth() == 1 && a.rows() == 1 && a.cols() == 1) {
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      float val = a.data[0][0][0];
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            float original = b.data[d][r][c];
            newData[d][r][c] = reversed ? (original - val) : (val - original);
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 2. Broadcast Matrix Subtraction (1xRxC) from (DxRxC)
    if (a.depth() == 1 && a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++) {
        for (int r = 0; r < b.rows(); r++) {
          for (int c = 0; c < b.cols(); c++) {
            float valA = a.data[0][r][c];
            float valB = b.data[d][r][c];
            newData[d][r][c] = reversed ? (valB - valA) : (valA - valB);
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // 3. Element-wise
    if (a.depth() == b.depth() && a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][][] newData = new float[depth()][rows()][cols()];
      for (int d = 0; d < depth(); d++) {
        for (int r = 0; r < rows(); r++) {
          for (int c = 0; c < cols(); c++) {
            // If reversed was true, 'b' is 'this' (the big one), 'a' is 'other' (small one logic, but here equal)
            // Actually if equal size, 'reversed' logic above handles strict pointer assignment.
            // Let's rely on original args for clarity in equal size:
            float v1 = this.data[d][r][c];
            float v2 = other.data[d][r][c];
            newData[d][r][c] = v1 - v2;
          }
        }
      }
      return new NumpyArray(newData);
    }
    throw new RuntimeException("Couldn't find subtract for " + dimension() + " / " + other.dimension());
  }
  
  public NumpyArray multiply(float value) {
    float[][][] newData = new float[depth()][rows()][cols()];
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          newData[d][r][c] = data[d][r][c] * value;
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  // Transpose: In 3D contexts (Neural Nets), transpose usually implies swapping 
  // the last two dimensions (Row/Col) while keeping Depth (Batch) intact.
  public NumpyArray transpose() {
    float[][][] newData = new float[depth()][cols()][rows()];
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          newData[d][c][r] = data[d][r][c];
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray inverse() {
    float[][][] newData = new float[depth()][rows()][cols()];
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          newData[d][r][c] = -data[d][r][c];
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  /*
   * Element-wise Multiplication (Hadamard Product)
   */
  public NumpyArray multiply(NumpyArray other) {
    NumpyArray a = size() < other.size() ? this : other;
    NumpyArray b = size() < other.size() ? other : this;
    
    // ====================================================
    // 1. Scalar multiply (1×1×1)
    // ====================================================
    if (a.size() == 1) {
      float val = a.data[0][0][0];
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++)
        for (int r = 0; r < b.rows(); r++)
          for (int c = 0; c < b.cols(); c++)
            newData[d][r][c] = val * b.data[d][r][c];
      
      return new NumpyArray(newData);
    }
    
    // ====================================================
    // 2. Elementwise exact match
    // ====================================================
    if (a.depth() == b.depth() &&
        a.rows() == b.rows() &&
        a.cols() == b.cols()) {
      
      float[][][] newData = new float[depth()][rows()][cols()];
      for (int d = 0; d < depth(); d++)
        for (int r = 0; r < rows(); r++)
          for (int c = 0; c < cols(); c++)
            newData[d][r][c] = a.data[d][r][c] * b.data[d][r][c];
      
      return new NumpyArray(newData);
    }
    
    // ====================================================
    // 3. Broadcast a (1×R×C) over b (D×R×C)
    // ====================================================
    if (a.depth() == 1 &&
        a.rows() == b.rows() &&
        a.cols() == b.cols()) {
      
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++)
        for (int r = 0; r < b.rows(); r++)
          for (int c = 0; c < b.cols(); c++)
            newData[d][r][c] = a.data[0][r][c] * b.data[d][r][c];
      
      return new NumpyArray(newData);
    }
    
    // ====================================================
    // 4. Broadcast row vector (1×1×C) → (1×R×C)
    //      Old: (1×C) × (R×C)
    // ====================================================
    if (a.depth() == 1 &&
        a.rows() == 1 &&
        a.cols() == b.cols()) {
      
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++)
        for (int r = 0; r < b.rows(); r++)
          for (int c = 0; c < b.cols(); c++)
            newData[d][r][c] = a.data[0][0][c] * b.data[d][r][c];
      
      return new NumpyArray(newData);
    }
    
    // ====================================================
    // 5. Broadcast column vector (1×R×1) → (1×R×C)
    //      Old: (R×1) × (R×C)
    // ====================================================
    if (a.depth() == 1 &&
        a.rows() == b.rows() &&
        a.cols() == 1) {
      
      float[][][] newData = new float[b.depth()][b.rows()][b.cols()];
      for (int d = 0; d < b.depth(); d++)
        for (int r = 0; r < b.rows(); r++)
          for (int c = 0; c < b.cols(); c++)
            newData[d][r][c] = a.data[0][r][0] * b.data[d][r][c];
      
      return new NumpyArray(newData);
    }
    
    throw new RuntimeException(
        "Dimension mismatch for multiply: " +
            a.dimension() + ", " + b.dimension()
    );
  }
  
  public NumpyArray divide(NumpyArray other) {
    // Simplified element-wise division logic for brevity, mirrors multiply/add structure
    // Assumes strict match or simple broadcasting
    if (this.depth() == other.depth() && this.rows() == other.rows() && this.cols() == other.cols()) {
      float[][][] newData = new float[depth()][rows()][cols()];
      for (int d = 0; d < depth(); d++) {
        for (int r = 0; r < rows(); r++) {
          for (int c = 0; c < cols(); c++) {
            newData[d][r][c] = this.data[d][r][c] / other.data[d][r][c];
          }
        }
      }
      return new NumpyArray(newData);
    }
    throw new RuntimeException("Strict dimension match required for divide currently: " + dimension() + " / " + other.dimension());
  }
  
  public NumpyArray randomize(float from, float to) {
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          data[d][r][c] = ThreadLocalRandom.current().nextFloat(from, to);
        }
      }
    }
    return this;
  }
  
  public NumpyArray flatten() {
    int total = depth() * rows() * cols();
    // Flatten to (1, total, 1) or (1, 1, total)? 
    // Original flattened to (total, 1). Let's keep that logic in 2nd/3rd dim.
    // Let's do (1, total, 1) to simulate a column vector.
    NumpyArray flat = new NumpyArray(1, total, 1);
    
    int counter = 0;
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          flat.data[0][counter++][0] = data[d][r][c];
        }
      }
    }
    return flat;
  }
  
  public float[] flattenArray() {
    float[] flat = new float[depth() * rows() * cols()];
    int counter = 0;
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          flat[counter++] = data[d][r][c];
        }
      }
    }
    return flat;
  }
  
  public NumpyArray reshape(int newDepth, int newRows, int newCols) {
    if (newDepth * newRows * newCols != size())
      throw new IllegalArgumentException("Invalid reshape dimensions: " + size() + " != " + (newDepth * newRows * newCols));
    
    float[][][] reshaped = new float[newDepth][newRows][newCols];
    
    // Linear mapping
    int oldCols = cols();
    int oldRows = rows();
    int oldSliceSize = oldCols * oldRows;
    
    int newSliceSize = newRows * newCols;
    
    for (int i = 0; i < size(); i++) {
      // Map linear 'i' to old (d, r, c)
      int oldD = i / oldSliceSize;
      int remOld = i % oldSliceSize;
      int oldR = remOld / oldCols;
      int oldC = remOld % oldCols;
      
      // Map linear 'i' to new (d, r, c)
      int newD = i / newSliceSize;
      int remNew = i % newSliceSize;
      int newR = remNew / newCols;
      int newC = remNew % newCols;
      
      reshaped[newD][newR][newC] = data[oldD][oldR][oldC];
    }
    return new NumpyArray(reshaped);
  }
  
  @Override
  public String toString() {
    return Arrays.deepToString(data);
  }
  
  public NumpyArray copy() {
    // Deep copy of 3D array
    float[][][] newData = new float[depth()][rows()][cols()];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        newData[i][j] = data[i][j].clone();
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray forAll(WeightInitializer supplier) {
    float[][][] newData = new float[depth()][rows()][cols()];
    for (int d = 0; d < depth(); d++) {
      for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
          newData[d][r][c] = supplier.get();
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  public String shape() {
    return "(" + depth() + ", " + rows() + ", " + cols() + ")";
  }
}