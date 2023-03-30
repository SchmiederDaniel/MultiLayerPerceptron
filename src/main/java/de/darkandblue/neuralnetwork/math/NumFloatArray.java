package de.darkandblue.neuralnetwork.math;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class NumFloatArray {
  final public float[][] data;
  
  @Deprecated
  public NumFloatArray(double[][] data) {
    float[][] newData = new float[data.length][data[0].length];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = (float) data[rowIndex][colIndex];
      }
    }
    this.data = newData;
  }
  
  public NumFloatArray(float[][] data) {
    this.data = data;
  }
  
  public NumFloatArray(int rows, int cols) {
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
  
  public static NumFloatArray identity(int n) {
    float[][] newData = new float[n][n];
    for (int i = 0; i < newData.length; i++) {
      newData[i][i] = 1;
    }
    
    return new NumFloatArray(newData);
  }
  
  public void checkMatchRows(NumFloatArray other) {
    if (rows() != other.rows()) {
      throw new IllegalArgumentException("Rows doesn't match " + rows() + ", " + other.rows());
    }
  }
  
  public void checkMatchCols(NumFloatArray other) {
    if (cols() != other.cols()) {
      throw new IllegalArgumentException("Cols doesn't match " + cols() + ", " + other.cols());
    }
  }
  
  /*
  Does Matrix multiplication with arrays
   */
  public NumFloatArray dot(NumFloatArray other) {
    if (cols() != other.rows()) {
      throw new RuntimeException("Columns doesn't match rows " + cols() + ", " + other.rows());
    }
    float[][] newData = new float[rows()][other.cols()];
    
    // slow
//    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
//      for (int colIndex = 0; colIndex < other.cols(); colIndex++) {
//        float sum = 0;
//        for (int otherColIndex = 0; otherColIndex < cols(); otherColIndex++) {
//          sum += data[rowIndex][otherColIndex] * other.data[otherColIndex][colIndex];
//        }
//        newData[rowIndex][colIndex] = sum;
//      }
//    }
    
    // fast
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int otherColIndex = 0; otherColIndex < cols(); otherColIndex++) {
        for (int colIndex = 0; colIndex < other.cols(); colIndex++) {
          newData[rowIndex][colIndex] += data[rowIndex][otherColIndex] * other.data[otherColIndex][colIndex];
        }
      }
    }
    
    return new NumFloatArray(newData);
  }
  
  public static NumFloatArray of(float... input) {
    float[][] newData = new float[input.length][1];
    for (int index = 0; index < input.length; index++) {
      newData[index][0] = input[index];
    }
    return new NumFloatArray(newData);
  }
  
  public NumFloatArray add(NumFloatArray other) {
    NumFloatArray a = rows() < other.rows() ? this : other;
    NumFloatArray b = rows() < other.rows() ? other : this;
    
    // add one number to every value in array
    if (a.rows() == 1 && a.cols() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] + b.data[rowIndex][colIndex];
        }
      }
      return new NumFloatArray(newData);
    }
    
    // add 1 dimensional array to a two-dimensional array
    if (a.rows() == 1 && a.cols() == b.cols()) {
      float[][] newData = new float[b.rows()][a.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < a.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] + b.data[rowIndex][colIndex];
        }
      }
      return new NumFloatArray(newData);
    }
    
    // matrix addition
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols(); coldIndex++) {
          newData[rowIndex][coldIndex] = a.data[rowIndex][coldIndex] + b.data[rowIndex][coldIndex];
        }
      }
      return new NumFloatArray(newData);
    }
    
    throw new RuntimeException("Couldn't find a add( function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  public String dimension() {
    return "(" + rows() + ", " + cols() + ")";
  }
  
  public NumFloatArray subtract(NumFloatArray other) {
    // determines if the subtraction needs to be reversed later on
    boolean reversed = !(rows() < other.rows());
    NumFloatArray a = rows() < other.rows() ? this : other;
    NumFloatArray b = rows() < other.rows() ? other : this;
    
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
      return new NumFloatArray(newData);
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
      return new NumFloatArray(newData);
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
      return new NumFloatArray(newData);
    }
    throw new RuntimeException("Couldn't find a subtract( function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  public static NumFloatArray subtractScalar(float value, NumFloatArray NumFloatArray) {
    float[][] newData = new float[NumFloatArray.rows()][NumFloatArray.cols()];
    for (int rowIndex = 0; rowIndex < NumFloatArray.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < NumFloatArray.cols(); colIndex++) {
        newData[rowIndex][colIndex] = value - NumFloatArray.data[rowIndex][colIndex];
      }
    }
    return new NumFloatArray(newData);
  }
  
  public NumFloatArray multiplyScalar(float other) {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] * other;
      }
    }
    return new NumFloatArray(newData);
  }
  
  public NumFloatArray subtractScalar(float other) {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] - other;
      }
    }
    return new NumFloatArray(newData);
  }
  
  public NumFloatArray transpose() {
    float[][] newData = new float[cols()][rows()];
    for (int rowsIndex = 0; rowsIndex < rows(); rowsIndex++) {
      for (int colsIndex = 0; colsIndex < cols(); colsIndex++) {
        newData[colsIndex][rowsIndex] = data[rowsIndex][colsIndex];
      }
    }
    return new NumFloatArray(newData);
  }

//  public NumFloatArray subtract(NumFloatArray other) {
//    return (NumFloatArray) super.subtract(other);
//  }
  
  public NumFloatArray inverse() {
    float[][] newData = new float[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = -data[rowIndex][colIndex];
      }
    }
    return new NumFloatArray(newData);
  }
  
  /*
  Copies the array * array feature of Numpy
   */
  public NumFloatArray multiplyScalar(NumFloatArray other) {
    NumFloatArray a = rows() < other.rows() ? this : other;
    NumFloatArray b = rows() < other.rows() ? other : this;
    
    if (a.cols() == 1 && a.rows() == 1) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumFloatArray(newData);
    }
    
    if (a.rows() == 1 && a.rows() == b.rows()) {
      float[][] newData = new float[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumFloatArray(newData);
    }
    
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      float[][] newData = new float[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[rowIndex][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumFloatArray(newData);
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
        
        return new NumFloatArray(newData);
      }
    }
    
    throw new RuntimeException("One array must be in the dimension of [1][x] or the same dimension of the second array " + a.dimension() + ", " + b.dimension());
  }
  
  public NumFloatArray randomize(float from, float to) {
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        data[rowIndex][colIndex] = ThreadLocalRandom.current().nextFloat(from, to);
      }
    }
    
    return this;
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
  
  public NumFloatArray copy() {
    float[][] newData = Arrays.stream(data).map(float[]::clone).toArray(float[][]::new);
    return new NumFloatArray(newData);
  }
}