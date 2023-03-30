package de.darkandblue.neuralnetwork;

import java.lang.reflect.Array;
import java.util.Arrays;

public class MatrixNDim {
  public static void main(String[] args) {
//    Object[] array = createArray(3, 4, 2, 3);
    Object array = Array.newInstance(float.class, 3, 4, 2, 3);
  
    System.out.println("" + Arrays.deepToString((Object[]) array));
  }
  
  static Object[] createArray(int... dimensions) {
    Object[] array = new Object[dimensions[0]];
    
    for (int i = 0; i < dimensions[0]; i++) {
      if(dimensions.length == 1) {
        array[i] = 0d;
      } else {
        array[i] = createArray(Arrays.copyOfRange(dimensions, 1, dimensions.length));
      }
    }
    
    return array;
  }
}