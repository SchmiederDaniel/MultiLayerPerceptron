package neuralnetwork.layer;

import neuralnetwork.initialization.Distribution;
import neuralnetwork.math.NumpyArray;
import neuralnetwork.math.NumpyArrayOld;


/**
 * 2D Convolution layer with stride.
 * <p>
 * Input: (1, H, W)
 * Weights: (filters, kH, kW)
 * Bias: (filters, 1, 1)
 * Output: (filters, outH, outW)
 */
public class Conv2D extends Layer {
  
  public NumpyArray weights;   // (filters, kH, kW)
  public NumpyArray bias;      // (filters, 1, 1)
  
  private NumpyArray input;    // stored for backward pass
  
  private final int filters;
  private final int kernelH, kernelW;
  private final int strideH, strideW;
  
  public Conv2D(int filters, int kernelH, int kernelW, int strideH, int strideW, Distribution distribution) {
    this(filters, kernelH, kernelW, strideH, strideW, new NumpyArray(filters, kernelH, kernelW), new NumpyArray(filters, 1, 1), null);
    
    distribution.setInputSize(kernelH * kernelW);
    distribution.setOutputSize(filters);
    
    weights = weights.forAll(distribution::randomWeight);
    bias = bias.forAll(distribution::randomBias);
  }
  
  public Conv2D(int filters, int kernelH, int kernelW, int strideH, int strideW, NumpyArray weights, NumpyArray bias, NumpyArray input) {
    this.filters = filters;
    this.kernelH = kernelH;
    this.kernelW = kernelW;
    this.strideH = strideH;
    this.strideW = strideW;
    this.weights = weights;
    this.bias = bias;
    this.input = input;
  }
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    if (input.depth() != 1)
      throw new IllegalArgumentException("Conv2D only supports single-depth inputs. got " + input.dimension());
    
    this.input = input;
    
    int H = input.rows();
    int W = input.cols();
    
    int outH = (H - kernelH) / strideH + 1;
    int outW = (W - kernelW) / strideW + 1;
    
    float[][][] out = new float[filters][outH][outW];
    
    for (int f = 0; f < filters; f++) {
      for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
          
          int inY = y * strideH;
          int inX = x * strideW;
          
          float sum = bias.data[f][0][0];
          
          for (int ky = 0; ky < kernelH; ky++) {
            for (int kx = 0; kx < kernelW; kx++) {
              float v = input.data[0][inY + ky][inX + kx];
              float w = weights.data[f][ky][kx];
              sum += v * w;
            }
          }
          
          out[f][y][x] = sum;
        }
      }
    }
    
    return new NumpyArray(out);
  }
  
  @Override
  public NumpyArray backward(NumpyArray dOut, float lr) {
    int H = input.rows();
    int W = input.cols();
    
    int outH = dOut.rows();
    int outW = dOut.cols();
    
    // Allocate gradients
    float[][][] dInput = new float[1][H][W];
    float[][][] dWeights = new float[filters][kernelH][kernelW];
    float[][][] dBias = new float[filters][1][1];
    
    // Compute gradients
    for (int f = 0; f < filters; f++) {
      for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
          
          float grad = dOut.data[f][y][x];
          
          int inY = y * strideH;
          int inX = x * strideW;
          
          // bias gradient
          dBias[f][0][0] += grad;
          
          // weight gradient
          for (int ky = 0; ky < kernelH; ky++) {
            for (int kx = 0; kx < kernelW; kx++) {
              float v = input.data[0][inY + ky][inX + kx];
              dWeights[f][ky][kx] += grad * v;
            }
          }
          
          // input gradient
          for (int ky = 0; ky < kernelH; ky++) {
            for (int kx = 0; kx < kernelW; kx++) {
              float w = weights.data[f][ky][kx];
              dInput[0][inY + ky][inX + kx] += grad * w;
            }
          }
        }
      }
    }
    
    // Update weights & biases
    for (int f = 0; f < filters; f++) {
      bias.data[f][0][0] -= lr * dBias[f][0][0];
      
      for (int ky = 0; ky < kernelH; ky++) {
        for (int kx = 0; kx < kernelW; kx++) {
          weights.data[f][ky][kx] -= lr * dWeights[f][ky][kx];
        }
      }
    }
    
    return new NumpyArray(dInput);
  }
  
  @Override
  public Layer deepCopy() {
    NumpyArray input = this.input == null ? null : this.input.copy();
    Conv2D c = new Conv2D(filters, kernelH, kernelW, strideH, strideW, weights.copy(), bias.copy(), input);
    return c;
  }
  
  /**
   * Returns the output dimension for an input of shape (1, inH, inW).
   * Format: { filters, outH, outW }
   */
  public int[] outputDimension(int inputHeight, int inputWidth) {
    int outH = (inputHeight - kernelH) / strideH + 1;
    int outW = (inputWidth - kernelW) / strideW + 1;
    return new int[]{filters, outH, outW};
  }
  
  @Override
  public String toString() {
    return "Conv2D(weights=" + weights.dimension() + ", bias=" + bias.dimension() + ", stride=(" + strideH + "," + strideW + "))";
  }
}