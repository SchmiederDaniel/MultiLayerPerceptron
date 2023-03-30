package de.darkandblue.neuralnetwork.benchmark;

import de.darkandblue.neuralnetwork.math.NumFloatArray;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import java.math.BigDecimal;

public class ArrayBenchmark {
  public static void main(String[] args) {
    int count = 500;
    System.out.println("start");
  
    NumFloatArray first = new NumFloatArray(784, 120);
    NumFloatArray second = new NumFloatArray(120, 784);
  
    NumFloatArray result = null;
    System.out.println("warmup");
    // warmup
    for (int i = 0; i < count / 2; i++) {
      result = first.dot(second);
    }
  
    System.out.println("benchmark");
    // benchmark
    long time = System.nanoTime();
    for (int i = 0; i < count; i++) {
      result = first.dot(second);
    }
    time = System.nanoTime() - time;
    System.out.println("time: " + new BigDecimal(time / (double) count).toPlainString() + "ns");
  }
}