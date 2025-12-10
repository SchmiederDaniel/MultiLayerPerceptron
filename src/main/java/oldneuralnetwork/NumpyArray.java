package oldneuralnetwork;

import oldneuralnetwork.initialization.WeightInitializer;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class NumpyArray {
  final public float[][] data;
  
  public NumpyArray(float[][] data) {
    this.data = data;
  }
  
  public NumpyArray(int rows, int cols) {
    this.data = new float[rows][cols];
  }
  
  /*
  Array is an implementation of the behavior of Numpy.array()
   */
  public int rows() { // Zeile
    return data.length;
  }
  
  public int cols() { // Spalte
    return data[0].length;
  }
  
  public static NumpyArray identity(int n) {
    float[][] newData = new float[n][n];
    for (int i = 0; i < newData.length; i++) {
      newData[i][i] = 1;
    }
    
    return new NumpyArray(newData);
  }
  
  public void checkMatchRows(NumpyArray other) {
    if (rows() != other.rows()) {
      throw new IllegalArgumentException("Rows doesn't match " + rows() + ", " + other.rows());
    }
  }
  
  public void checkMatchCols(NumpyArray other) {
    if (cols() != other.cols()) {
      throw new IllegalArgumentException("Cols doesn't match " + cols() + ", " + other.cols());
    }
  }
  
  /*
  Does Matrix multiplication with arrays
   */
  public NumpyArray dot(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    if (a.cols() == 1 && a.rows() == 1) { // matrix * number
      NumpyArray b = rows() < other.rows() ? other : this;
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    if (cols() != other.rows()) {
      throw new RuntimeException("Columns doesn't match rows " + cols() + ", " + other.rows());
    }
    // Matrix multiplication (matrix * matrix)
    float[][] newData = new float[rows()][other.cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int otherColIndex = 0; otherColIndex < cols(); otherColIndex++) {
        for (int colIndex = 0; colIndex < other.cols(); colIndex++) {
          newData[rowIndex][colIndex] += data[rowIndex][otherColIndex] * other.data[otherColIndex][colIndex];
        }
      }
    }
    return new NumpyArray(newData);
  }
  
  public static NumpyArray of(float... input) {
    float[][] newData = new float[input.length][1];
    for (int index = 0; index < input.length; index++) {
      newData[index][0] = input[index];
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray add(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    // add one number to every value in array
    if (a.rows() == 1 && a.cols() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] + b.data[rowIndex][colIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    // add 1 dimensional array to a two-dimensional array
    if (a.rows() == 1 && a.cols() == b.cols()) {
      float[][] newData = new float[b.rows()][a.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < a.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] + b.data[rowIndex][colIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    // matrix addition
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols(); coldIndex++) {
          newData[rowIndex][coldIndex] = a.data[rowIndex][coldIndex] + b.data[rowIndex][coldIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    throw new RuntimeException("Couldn't find a add( function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  public String dimension() {
    return "(" + rows() + ", " + cols() + ")";
  }
  
  public NumpyArray subtract(NumpyArray other) {
    // reverse when other has more rows (is a matrix)
    boolean reversed = !(rows() < other.rows());
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    // subtract one number to every value in array
    if (a.rows() == 1 && a.cols() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
          if (reversed) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] - a.data[0][0];
          } else {
            newData[rowIndex][colIndex] = a.data[0][0] - b.data[rowIndex][colIndex];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // subtract 1 dimensional array from a two-dimensional array
    if (a.rows() == 1 && a.cols() == b.cols()) {
      float[][] newData = new float[b.rows()][a.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < a.cols(); colIndex++) {
          if (reversed) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] - a.data[0][colIndex];
          } else {
            newData[rowIndex][colIndex] = a.data[0][colIndex] - b.data[rowIndex][colIndex];
          }
        }
      }
      return new NumpyArray(newData);
    }
    
    // matrix subtraction
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols(); coldIndex++) {
          if (reversed) {
            newData[rowIndex][coldIndex] = b.data[rowIndex][coldIndex] - a.data[rowIndex][coldIndex];
          } else {
            newData[rowIndex][coldIndex] = a.data[rowIndex][coldIndex] - b.data[rowIndex][coldIndex];
          }
        }
      }
      return new NumpyArray(newData);
    }
    throw new RuntimeException("Couldn't find a subtract( function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  public static NumpyArray subtractScalar(float value, NumpyArray numpyArray) {
    float[][] newData = new float[numpyArray.rows()][numpyArray.cols()];
    for (int rowIndex = 0; rowIndex < numpyArray.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < numpyArray.cols(); colIndex++) {
        newData[rowIndex][colIndex] = value - numpyArray.data[rowIndex][colIndex];
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray multiply(float other) {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] * other;
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray subtractScalar(float other) {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] - other;
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray transpose() {
    float[][] newData = new float[cols()][rows()];
    for (int rowsIndex = 0; rowsIndex < rows(); rowsIndex++) {
      for (int colsIndex = 0; colsIndex < cols(); colsIndex++) {
        newData[colsIndex][rowsIndex] = data[rowsIndex][colsIndex];
      }
    }
    return new NumpyArray(newData);
  }

//  public NumpyArray subtract(NumpyArray other) {
//    return (NumpyArray) super.subtract(other);
//  }
  
  public NumpyArray inverse() {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = -data[rowIndex][colIndex];
      }
    }
    return new NumpyArray(newData);
  }
  
  
  /*
  Represents the array * array feature of Numpy
   */
  public NumpyArray multiply(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    if (a.cols() == 1 && a.rows() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if (a.rows() == 1 && a.rows() == b.rows()) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    // Matrix scalar multiplication matrix[x][y] * matrix[x][y]
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[rowIndex][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if (a.rows() == b.rows()) {
      a = cols() < other.cols() ? this : other;
      b = cols() < other.cols() ? other : this;
      
      if (a.cols() == 1) {
        float[][] newData = new float[rows()][b.cols()];
        for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
          for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] * a.data[rowIndex][0];
          }
        }
        
        return new NumpyArray(newData);
      }
    }
    
    throw new RuntimeException("One array must be in the dimension of [1][x] or the same dimension of the second array " + a.dimension() + ", " + b.dimension());
  }
  
  public NumpyArray divide(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    if (a.cols() == 1 && a.rows() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] / b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if (a.rows() == 1 && a.rows() == b.rows()) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] / b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    // Matrix scalar multiplication matrix[x][y] * matrix[x][y]
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[rowIndex][colIndex] / b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if (a.rows() == b.rows()) {
      a = cols() < other.cols() ? this : other;
      b = cols() < other.cols() ? other : this;
      
      if (a.cols() == 1) {
        float[][] newData = new float[rows()][b.cols()];
        for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
          for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] / a.data[rowIndex][0];
          }
        }
        
        return new NumpyArray(newData);
      }
    }
    
    throw new RuntimeException("One array must be in the dimension of [1][x] or the same dimension of the second array " + a.dimension() + ", " + b.dimension());
  }
  
  public NumpyArray randomize(float from, float to) {
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        data[rowIndex][colIndex] = ThreadLocalRandom.current().nextFloat(from, to);
      }
    }
    
    return this;
  }
  
  public NumpyArray flatten() {
    int cols = cols();
    int total = rows() * cols;
    NumpyArray flat = new NumpyArray(total, 1);
    for (int i = 0; i < rows(); i++)
      for (int j = 0; j < cols; j++)
        flat.data[i * cols + j][0] = data[i][j];
    return flat;
  }
  
  public float[] flattenArray() {
    float[] flat = new float[rows() * cols()];
    for (int i = 0; i < rows(); i++)
      for (int j = 0; j < cols(); j++)
        flat[i * cols() + j] = data[i][j];
    return flat;
  }
  
  public NumpyArray reshape(int newRows, int newCols) {
    int cols = cols();
    if (newRows * newCols != rows() * cols)
      throw new IllegalArgumentException("Invalid reshape dimensions");
    NumpyArray reshaped = new NumpyArray(newRows, newCols);
    for (int i = 0; i < newRows * newCols; i++) {
      reshaped.data[i / newCols][i % newCols] = data[i / cols][i % cols];
    }
    return reshaped;
  }
  
  
  @Override
  public String toString() {
//    String output = "[";
//    
//    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
//      output += "[";
//      for (int colIndex = 0; colIndex < data[rowIndex].length; colIndex++) {
//        output += new BigDecimal(data[rowIndex][colIndex]).setScale(2, RoundingMode.HALF_UP).toPlainString() + ", ";
//      }
//      output = output.substring(0, output.length() - 2);
//      output += "], ";
//    }
//    output = output.substring(0, output.length() - 2);
//    
//    return output + "]";
    return Arrays.deepToString(data);
  }
  
  public NumpyArray copy() {
    float[][] newData = Arrays.stream(data).map(float[]::clone).toArray(float[][]::new);
    return new NumpyArray(newData);
  }
  
  public NumpyArray forAll(WeightInitializer supplier) {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        Float value = supplier.get();
        newData[rowIndex][colIndex] = value;
      }
    }
    return new NumpyArray(newData);
  }
}