package neuralnetwork.layer;

import neuralnetwork.math.NumpyArray;

public class Reshape extends Layer {

  private final int newDepth;
  private final int newRows;
  private final int newCols;

  private int oldDepth;
  private int oldRows;
  private int oldCols;

  private NumpyArray input;

  public Reshape(int newDepth, int newRows, int newCols) {
    this.newDepth = newDepth;
    this.newRows = newRows;
    this.newCols = newCols;
  }

  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;

    oldDepth = input.depth();
    oldRows = input.rows();
    oldCols = input.cols();

    int newSize = newDepth * newRows * newCols;
    if (newSize != input.size())
      throw new IllegalArgumentException(
          "Cannot reshape from " + input.dimension() +
          " to ("+ newDepth +", "+ newRows +", "+ newCols +")");
    
    NumpyArray reshape = input.reshape(newDepth, newRows, newCols);
//    System.out.println("Reshaped had to: (" + newDepth + ", " + newRows + ", " + newCols + ") from " + input.dimension() + " to " + reshape.dimension());
    return reshape;
  }

  @Override
  public NumpyArray backward(NumpyArray outputGradient, float learningRate) {
    // Reverse the reshape
    return outputGradient.reshape(oldDepth, oldRows, oldCols);
  }

  @Override
  public Layer deepCopy() {
    Reshape r = new Reshape(newDepth, newRows, newCols);
    r.oldDepth = this.oldDepth;
    r.oldRows = this.oldRows;
    r.oldCols = this.oldCols;
    r.input = (input == null ? null : input.copy());
    return r;
  }

  @Override
  public String toString() {
    return "Reshape -> (" + newDepth + ", " + newRows + ", " + newCols + ")";
  }
}