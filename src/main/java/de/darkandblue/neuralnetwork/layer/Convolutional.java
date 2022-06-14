package de.darkandblue.neuralnetwork.layer;

import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Convolutional extends Layer {
  int depth;
  int input_depth;
  int[] input_shape;
  int[] output_shape;
  int[] kernels_shape;
  NumpyArray kernels;
  NumpyArray biases;
  NumpyArray input;
  
  /*
  input_shape contains input_depth, input_height, input_width which determine the shape of the convolutional layer
   */
  public Convolutional(int[] input_shape, int kernel_size, int depth) {
    this.depth = depth;
    this.input_shape = input_shape;
    this.input_depth = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    this.output_shape = new int[] {
      depth, input_height - kernel_size + 1, input_width - kernel_size + 1
    };
    this.kernels_shape = new int[] {
      depth, input_depth, kernel_size, kernel_size
    };
    
//    this.kernels = NumpyArray.createFromDimensions(kernels_shape);
  }
  
  @Override
  public NumpyArray forward(NumpyArray input) {
//    this.input = input;
//    NumpyArray output = biases.copy();
//    for (int i = 0; i < depth; i++) {
//      for j in range(self.input_depth) {
//        self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j],"valid")
//      }
//    }
//    return self.output
    return null;
  }
  
  @Override
  public NumpyArray backward(NumpyArray output_gradient, double learning_rate) {
    return null;
  }
}