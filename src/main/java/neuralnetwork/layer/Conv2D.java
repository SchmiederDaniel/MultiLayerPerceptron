package neuralnetwork.layer;

import neuralnetwork.initialization.Distribution;
import neuralnetwork.math.NumpyArray;
import neuralnetwork.math.NumpyArrayOld;

import java.util.Arrays;

/**
 * Multi-channel 2D Convolution layer.
 * <p>
 * Input:  (C_in, H, W)
 * Weights: filters × (C_in, kH, kW)
 * Bias:   (filters, 1, 1)
 * Output: (filters, outH, outW)
 */
public class Conv2D extends Layer {
  
  private final int inChannels;
  private final int filters;
  private final int kernelH, kernelW;
  private final int strideH, strideW;
  
  public NumpyArray[] weights; // each weights[f] has shape (C_in, kH, kW)
  public NumpyArray bias;      // (filters, 1, 1)
  
  private NumpyArray input;    // store full multi-channel input
  
  public Conv2D(int inChannels, int filters, int kernelH, int kernelW,
                int strideH, int strideW, Distribution dist) {
    this.inChannels = inChannels;
    this.filters = filters;
    this.kernelH = kernelH;
    this.kernelW = kernelW;
    this.strideH = strideH;
    this.strideW = strideW;
    
    // Allocate weights
    weights = new NumpyArray[filters];
    for (int f = 0; f < filters; f++) {
      weights[f] = new NumpyArray(inChannels, kernelH, kernelW);
    }
    
    bias = new NumpyArray(filters, 1, 1);
    
    dist.setInputSize(inChannels * kernelH * kernelW);
    dist.setOutputSize(filters);
    
    // init weights
    for (int f = 0; f < filters; f++)
      weights[f] = weights[f].forAll(dist::randomWeight);
    
    bias = bias.forAll(dist::randomBias);
  }
  
  
  public Conv2D(int inChannels, int filters, int kernelH, int kernelW, int strideH, int strideW, NumpyArray[] weights, NumpyArray bias, NumpyArray input) {
    this.inChannels = inChannels;
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
    
    if (input.depth() != inChannels)
      throw new IllegalArgumentException(
          "Conv2D expected " + inChannels + " channels, got " + input.depth()
      );
    
    this.input = input;
    
    int H = input.rows();
    int W = input.cols();
    
    int outH = (H - kernelH) / strideH + 1;
    int outW = (W - kernelW) / strideW + 1;
    
    float[][][] out = new float[filters][outH][outW];
    
    // Convolution
    for (int f = 0; f < filters; f++) {
      NumpyArray Wf = weights[f];
      
      for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
          
          float sum = bias.data[f][0][0];
          int inY = y * strideH;
          int inX = x * strideW;
          
          for (int c = 0; c < inChannels; c++) {
            for (int ky = 0; ky < kernelH; ky++) {
              for (int kx = 0; kx < kernelW; kx++) {
                float v = input.data[c][inY + ky][inX + kx];
                float w = Wf.data[c][ky][kx];
                sum += v * w;
              }
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
    
    float[][][] dInput = new float[inChannels][H][W];
    float[][][][] dWeights = new float[filters][inChannels][kernelH][kernelW];
    float[][][] dBias = new float[filters][1][1];
    
    // Backprop
    for (int f = 0; f < filters; f++) {
      for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
          
          float grad = dOut.data[f][y][x];
          dBias[f][0][0] += grad;
          
          int inY = y * strideH;
          int inX = x * strideW;
          
          for (int c = 0; c < inChannels; c++) {
            for (int ky = 0; ky < kernelH; ky++) {
              for (int kx = 0; kx < kernelW; kx++) {
                
                // weight grad
                dWeights[f][c][ky][kx] +=
                    input.data[c][inY + ky][inX + kx] * grad;
                
                // input grad
                dInput[c][inY + ky][inX + kx] +=
                    weights[f].data[c][ky][kx] * grad;
              }
            }
          }
        }
      }
    }
    
    // Apply gradient update
    for (int f = 0; f < filters; f++) {
      bias.data[f][0][0] -= lr * dBias[f][0][0];
      
      for (int c = 0; c < inChannels; c++) {
        for (int ky = 0; ky < kernelH; ky++) {
          for (int kx = 0; kx < kernelW; kx++) {
            weights[f].data[c][ky][kx] -= lr * dWeights[f][c][ky][kx];
          }
        }
      }
    }
    
    return new NumpyArray(dInput);
  }
  
  @Override
  public Layer deepCopy() {
    NumpyArray input = this.input == null ? null : this.input.copy();
    NumpyArray weightsCopy[] = Arrays.stream(weights).map(NumpyArray::copy).toArray(NumpyArray[]::new);
    Conv2D c = new Conv2D(
        inChannels,
        filters,
        kernelH, kernelW,
        strideH, strideW,
        weightsCopy,
        bias.copy(),
        input
    );
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
  
}