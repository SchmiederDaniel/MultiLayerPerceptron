package de.darkandblue.neuralnetwork.math;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class NumpyArray {
  final public double[][] data;
  
  public NumpyArray(double[][] data) {
    this.data = data;
  }
  
  public NumpyArray(int rows, int cols) {
    this.data = new double[rows][cols];
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
    double[][] newData = new double[n][n];
    for (int i = 0; i < newData.length; i++) {
      newData[i][i] = 1;
    }
    
    return new NumpyArray(newData);
  }
  
  /*
  Does Matrix multiplication with arrays
   */
  public NumpyArray dot(NumpyArray other) {
    if (cols() != other.rows()) {
      throw new RuntimeException("Columns doesn't match rows " + cols() + ", " + other.rows());
    }
    double[][] newData = new double[rows()][other.cols()];
  
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < other.cols(); colIndex++) {
        double sum = 0;
        for (int otherColIndex = 0; otherColIndex < cols(); otherColIndex++) {
          sum += data[rowIndex][otherColIndex] * other.data[otherColIndex][colIndex];
        }
        newData[rowIndex][colIndex] = sum;
      }
    }
  
    return new NumpyArray(newData);
  }
  
  public static NumpyArray of(double... input) {
    double[][] newData = new double[input.length][1];
    for (int index = 0; index < input.length; index++) {
      newData[index][0] = input[index];
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray add(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    // add one number to every value in array
    if(a.rows() == 1 && a.cols() == 1) {
      double[][] newData = new double[b.rows()][b.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] + b.data[rowIndex][colIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    // add 1 dimensional array to a two dimensional array
    if (a.rows() == 1 && a.cols() == b.cols()) {
      double[][] newData = new double[b.rows()][a.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < a.cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] + b.data[rowIndex][colIndex];
        }
      }
      return new NumpyArray(newData);
    }
    
    // matrix addition
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      double[][] newData = new double[rows()][cols()];
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
    // determines if the subtraction needs to be reversed later on
    boolean reversed = !(rows() < other.rows());
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
  
    // subtract one number to every value in array
    if(a.rows() == 1 && a.cols() == 1) {
      double[][] newData = new double[b.rows()][b.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
          if(reversed) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] - a.data[0][0];
          } else {
            newData[rowIndex][colIndex] = a.data[0][0] - b.data[rowIndex][colIndex];
          }
        }
      }
      return new NumpyArray(newData);
    }
  
    // subtract 1 dimensional array from a two dimensional array
    if (a.rows() == 1 && a.cols() == b.cols()) {
      double[][] newData = new double[b.rows()][a.cols()];
      for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < a.cols(); colIndex++) {
          if(reversed) {
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
      double[][] newData = new double[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols(); coldIndex++) {
          if(reversed) {
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
  
  public static NumpyArray subtractScalar(double value, NumpyArray numpyArray) {
    double[][] newData = new double[numpyArray.rows()][numpyArray.cols()];
    for (int rowIndex = 0; rowIndex < numpyArray.rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < numpyArray.cols(); colIndex++) {
        newData[rowIndex][colIndex] = value - numpyArray.data[rowIndex][colIndex];
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray multiplyScalar(double other) {
    double[][] newData = new double[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] * other;
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray subtractScalar(double other) {
    double[][] newData = new double[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = data[rowIndex][colIndex] - other;
      }
    }
    return new NumpyArray(newData);
  }
  
  public NumpyArray transpose() {
    double[][] newData = new double[cols()][rows()];
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
    double[][] newData = new double[rows()][cols()];
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        newData[rowIndex][colIndex] = -data[rowIndex][colIndex];
      }
    }
    return new NumpyArray(newData);
  }
  
  /*
  Copies the array * array feature of Numpy
   */
  public NumpyArray multiplyScalar(NumpyArray other) {
    NumpyArray a = rows() < other.rows() ? this : other;
    NumpyArray b = rows() < other.rows() ? other : this;
    
    if(a.cols() == 1 && a.rows() == 1) {
      double[][] newData = new double[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
        }
      }
  
      return new NumpyArray(newData);
    }
    
    if (a.rows() == 1 && a.rows() == b.rows()) {
      double[][] newData = new double[b.rows()][b.cols()];
      for (int colIndex = 0; colIndex < b.cols(); colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows(); rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      double[][] newData = new double[rows()][cols()];
      for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
        for (int colIndex = 0; colIndex < cols(); colIndex++) {
          newData[rowIndex][colIndex] = a.data[rowIndex][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      
      return new NumpyArray(newData);
    }
    
    if(a.rows() == b.rows()) {
      a = cols() < other.cols() ? this : other;
      b = cols() < other.cols() ? other : this;
  
      if(a.cols() == 1) {
        double[][] newData = new double[rows()][b.cols()];
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
  
  public NumpyArray randomize(double from, double to) {
    for (int rowIndex = 0; rowIndex < rows(); rowIndex++) {
      for (int colIndex = 0; colIndex < cols(); colIndex++) {
        data[rowIndex][colIndex] = ThreadLocalRandom.current().nextDouble(from, to);
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
  
  public NumpyArray copy() {
    double[][] newData = Arrays.stream(data).map(double[]::clone).toArray(double[][]::new);
    return new NumpyArray(newData);
  }
}