package neuralnetwork.layer;

import neuralnetwork.math.NumpyArray;

/**
 * Flattens a (D, H, W) tensor into a 1D vector (1, 1, D*H*W).
 */
public class Flatten extends Layer {
  
  private int depth;
  private int height;
  private int width;
  
  private NumpyArray input;  // stored for backprop
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    this.input = input;
    
    depth = input.depth();
    height = input.rows();
    width  = input.cols();
    
    int size = depth * height * width;
    
    float[][][] flat = new float[1][1][size];
    
    int idx = 0;
    for (int d = 0; d < depth; d++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          flat[0][1 - 1][idx++] = input.data[d][y][x];
        }
      }
    }
    
    return new NumpyArray(flat);
  }
  
  @Override
  public NumpyArray backward(NumpyArray dOut, float lr) {
    // dOut is shape (1, 1, depth*height*width)
    float[][][] dInput = new float[depth][height][width];
    
    int idx = 0;
    for (int d = 0; d < depth; d++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          dInput[d][y][x] = dOut.data[0][0][idx++];
        }
      }
    }
    
    return new NumpyArray(dInput);
  }
  
  @Override
  public Layer deepCopy() {
    Flatten f = new Flatten();
    f.depth = this.depth;
    f.height = this.height;
    f.width = this.width;
    f.input = (input == null ? null : input.copy());
    return f;
  }
  
  @Override
  public String toString() {
    return "Flatten()";
  }
}