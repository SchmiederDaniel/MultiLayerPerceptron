package de.darkandblue.neuralnetwork.math;

import java.lang.reflect.Array;

public class NdArray {
  Object data;
  
  public NdArray(int... dimensions) {
    data = Array.newInstance(double.class, dimensions);
  }
  
  private static void deepToString(Object[] a) {
    if (a.length == 0) {
      throw new RuntimeException("Array has length 0");
    }
    
    for (Object element : a) {
      if (element == null) {
        throw new RuntimeException("Array contained null object");
      } else {
        Class<?> eClass = element.getClass();
        
        if (eClass.isArray()) {
          if (eClass == double[].class) {
            System.out.println("Do something with the one dimensional double array");
          } else { // element is an array of object references
            deepToString((Object[]) element);
          }
        } else {  // element is non-null and not an array
          throw new RuntimeException("Array contained non array object " + element + " " + element.getClass());
        }
      }
    }
  }
}